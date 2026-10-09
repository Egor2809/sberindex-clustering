"""Create an offline municipal map from the licensed registry GeoPackage.

Geometry is simplified for display only. It never enters model features.
Uses the standard GeoPackage binary header and Shapely, without GDAL.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import numpy as np
from shapely import coverage_invalid_edges, coverage_is_valid, coverage_simplify, from_wkb, make_valid
from shapely.ops import transform


def decode_geometry(blob):
    if blob[:2] != b'GP' or blob[2] != 0:
        raise ValueError('Not a version-0 GeoPackage geometry')
    flags = blob[3]
    if flags & 0b00110000:
        raise ValueError('Empty or extended GeoPackage geometry')
    envelope = (flags >> 1) & 7
    if envelope not in range(5):
        raise ValueError('Reserved envelope code')
    endian = '<' if flags & 1 else '>'
    if struct.unpack(endian+'i', blob[4:8])[0] != 4326:
        raise ValueError('Expected EPSG:4326 input')
    offset = 8 + 8 * (0, 4, 6, 6, 8)[envelope]
    return from_wkb(blob[offset:])


def albers(lon, lat):
    """Spherical equal-area Albers; parallels 50/70 N, meridian 100 E."""
    p1, p2, p0 = np.deg2rad([50., 70., 40.])
    n = (np.sin(p1) + np.sin(p2)) / 2
    c = np.cos(p1)**2 + 2*n*np.sin(p1)
    rho0 = np.sqrt(c-2*n*np.sin(p0))/n
    rho = np.sqrt(c-2*n*np.sin(np.deg2rad(lat)))/n
    theta = n*np.deg2rad(np.asarray(lon)-100)
    return rho*np.sin(theta), -(rho0-rho*np.cos(theta))


def _polygon_signature(geom):
    polygons = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    return len(polygons), tuple(len(polygon.interiors) for polygon in polygons)


def simplify_shared_boundaries(geometries, tolerance):
    """Simplify matched interior edges without inventing fixes for source overlaps.

    GeoPackage polygons with invalid coverage edges are kept unchanged. Outer edges of
    the valid subset are also kept unchanged, so their joins to quarantined polygons
    and all coastlines remain exactly as supplied by the registry.
    """
    keys = sorted(geometries)
    original = np.asarray([geometries[key] for key in keys], dtype=object)
    invalid_before = coverage_invalid_edges(original)
    quarantined = np.asarray([not edge.is_empty for edge in invalid_before], dtype=bool)
    simplified = original.copy()
    active = ~quarantined
    if active.any():
        if not coverage_is_valid(original[active]):
            raise ValueError('Non-quarantined municipal polygons are not an edge-matched coverage')
        simplified[active] = coverage_simplify(
            original[active], tolerance, simplify_boundary=False
        )
        if not coverage_is_valid(simplified[active]):
            raise ValueError('Shared-boundary simplification broke the valid municipal coverage')

    for key, before, after, is_quarantined in zip(keys, original, simplified, quarantined):
        if after.is_empty or not after.is_valid or after.geom_type not in ('Polygon', 'MultiPolygon'):
            raise ValueError('Invalid simplified geometry for ' + key)
        if _polygon_signature(after) != _polygon_signature(before):
            raise ValueError('Simplification changed polygon components or holes for ' + key)
        if is_quarantined and not after.equals_exact(before, 0):
            raise ValueError('Simplification changed quarantined source geometry for ' + key)

    invalid_after = coverage_invalid_edges(simplified)
    bad_before = {keys[i] for i, edge in enumerate(invalid_before) if not edge.is_empty}
    bad_after = {keys[i] for i, edge in enumerate(invalid_after) if not edge.is_empty}
    length_before = sum(edge.length for edge in invalid_before)
    length_after = sum(edge.length for edge in invalid_after)
    if bad_after != bad_before or not math.isclose(length_after, length_before, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Simplification changed the registry's pre-existing invalid coverage edges")

    result = {key: geom for key, geom in zip(keys, simplified)}
    audit = {
        'algorithm': 'GEOS coverage_simplify (Visvalingam-Whyatt), shared interior edges only',
        'tolerance_degrees': tolerance,
        'outer_boundaries_preserved': True,
        'source_coverage_valid': not bad_before,
        'quarantined_source_geometry_ids': sorted(bad_before),
        'quarantined_source_geometry_count': len(bad_before),
        'invalid_edge_length_degrees_before': length_before,
        'invalid_edge_length_degrees_after': length_after,
        'valid_subset_edge_matched_after': bool(coverage_is_valid(simplified[active])) if active.any() else True,
    }
    return result, audit


def build(gpkg, assignments, output, tolerance=.025):
    import csv
    with assignments.open(encoding='utf-8-sig') as stream:
        ids = {row['entity_id'] for row in csv.DictReader(stream)}
    if len(ids) != 2016:
        raise ValueError('Expected the frozen 2016-territory cohort')
    conn = sqlite3.connect(gpkg.resolve().as_uri()+'?mode=ro', uri=True)
    rows = conn.execute('SELECT territory_id,geom FROM t_dict_municipal_districts_poly WHERE year_from<=2023 AND year_to>2023')
    geometries = {}
    display_repairs = []
    for tid, blob in rows:
        key = 'tid_'+str(tid)
        if key not in ids:
            continue
        if key in geometries:
            raise ValueError('Ambiguous polygon for '+key)
        geom = decode_geometry(blob)
        # Unwrap the dateline before simplification/projection, keeping Chukotka continuous.
        geom = transform(lambda x,y,z=None:(np.where(np.asarray(x)<0,np.asarray(x)+360,x),y),geom)
        if geom.geom_type not in ('Polygon','MultiPolygon'):
            raise ValueError('Nonpolygon geometry for '+key)
        if not geom.is_valid:
            from shapely.validation import explain_validity
            reason = explain_validity(geom)
            geom = make_valid(geom)
            if geom.geom_type == 'GeometryCollection':
                from shapely.ops import unary_union
                geom = unary_union([g for g in geom.geoms if g.geom_type in ('Polygon','MultiPolygon')])
            if geom.is_empty or geom.geom_type not in ('Polygon','MultiPolygon') or not geom.is_valid:
                raise ValueError('Cannot render geometry for '+key)
            display_repairs.append({'id':key,'reason_after_dateline_unwrap':reason,'operation':'Shapely make_valid; polygon components only; display only'})
        geometries[key] = geom
    conn.close()
    geometries, topology_audit = simplify_shared_boundaries(geometries, tolerance)
    geometries = {key: transform(albers, geom) for key, geom in geometries.items()}
    bounds = np.array([g.bounds for g in geometries.values()])
    xmin,ymin = bounds[:,:2].min(axis=0)
    xmax,ymax = bounds[:,2:].max(axis=0)
    scale = min(970/(xmax-xmin),390/(ymax-ymin))
    offsetx = (1000-(xmax-xmin)*scale)/2
    offsety = (420-(ymax-ymin)*scale)/2
    paths = []
    for key, geom in sorted(geometries.items()):
        segments = []
        polys = [geom] if geom.geom_type=='Polygon' else list(geom.geoms)
        for polygon in polys:
            for ring in [polygon.exterior,*polygon.interiors]:
                points = np.array(ring.coords)
                points[:,0] = offsetx+(points[:,0]-xmin)*scale
                points[:,1] = offsety+(points[:,1]-ymin)*scale
                segments.append('M'+'L'.join(f'{x:.2f},{y:.2f}' for x,y in points[:-1])+'Z')
        paths.append({'id':key,'d':''.join(segments)})
    result = {'viewBox':[0,0,1000,420],'paths':paths,'matched':len(paths),
        'missing_ids':sorted(ids-set(geometries)),
        'source':'СберИндекс: реестр муниципалитетов, геометрия с year_from ≤ 2023 < year_to. CC BY-SA 4.0; исходные границы OpenStreetMap. Показан охват панели, а не вся страна.',
        'projection':'Spherical Albers equal area, parallels 50/70N, central meridian 100E',
        'simplification_degrees':tolerance,'topology_audit':topology_audit,
        'source_sha256':hashlib.sha256(gpkg.read_bytes()).hexdigest(),
        'display_repairs':display_repairs,
        'license':'CC BY-SA 4.0; OpenStreetMap attribution retained'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(json.dumps({'matched':len(paths),'missing':len(result['missing_ids']),'bytes':output.stat().st_size}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpkg',type=Path,required=True)
    p.add_argument('--assignments',type=Path,default=Path('reports/experiments/2026-09-23-v2/reference_assignments.csv'))
    p.add_argument('--output',type=Path,default=Path('reports/contest-v3/municipal_map.json'))
    a=p.parse_args()
    build(a.gpkg,a.assignments,a.output)
