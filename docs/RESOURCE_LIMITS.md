# Ограничение нагрузки при локальных расчётах

`scripts/run_bounded.py` служит опциональным supervisor для Windows. Он запускает один worker с пониженным приоритетом, помещает всё дерево процессов в Job Object и пишет журнал в `artifacts/resource_runs/`. Параметры машины не зашиты в код и не берутся из исследовательского конфига: их передают отдельным JSON-файлом через `--profile`.

Профиль содержит `cpu_hard_cap_percent`, `memory_limit_bytes`, `threads`, `system_monitor_ceiling_percent`, `recovery_ceiling_percent`, `sample_interval_seconds`, `active_seconds` и `pause_seconds`. Его следует подобрать под конкретную машину и хранить вне репозитория, например в исключённом из Git каталоге `.local/`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-monitor.txt
.\.venv\Scripts\python.exe scripts/run_bounded.py `
  --config configs/research_annual.json `
  --profile .local/runtime-profile.json `
  --timeout 3600 --wait-for-headroom 30 --pause-for-headroom 60
```

Job Object задаёт жёсткую квоту процессорного времени и суммарной private committed memory для worker и его дочерних процессов. Число потоков библиотек также ограничивается значением `threads`; выбранные методы запускаются без CUDA. Worker проверяет фактические CPU- и memory-параметры Job Object перед импортом численных библиотек.

Общесистемная загрузка CPU, RAM, GPU, видеопамяти и каждого физического диска измеряется с заданным интервалом. RAM считается как `(total - available) / total`: Windows включает освобождаемый standby/cache в `available`, поэтому файловый кэш сам по себе не считается давлением памяти. Активность каждого физического диска получается как `100 - % Idle Time` из Windows Performance Counters.

Превышение системного порога удерживает уже приостановленное дерево процессов. Оно возобновляется только после трёх последовательных безопасных измерений или завершается по таймауту. Задержка мониторинга также останавливает запуск. Закрытие Job Object завершает всё дерево worker, а незавершённый исследовательский run помечается как interrupted.

Дисковая метрика является сэмплируемым наблюдением, а не мгновенным аппаратным ограничением. Короткий всплеск между замерами возможен, а фоновые приложения могут вызвать паузу или остановку. В журнале поэтому фиксируются наблюдавшиеся значения и реакция supervisor; он не обещает жёсткий предел дисковой активности.

На Linux и macOS `scripts/run_research.py` и `scripts/run_contest.py` запускают расчёт напрямую и ограничивают потоки численных библиотек. Для системных CPU/memory/I/O-квот используйте cgroup, контейнер или планировщик среды. Windows-supervisor включается явно параметром `--supervisor PROFILE` у этих entrypoint.

Механизмы Windows: [ограничение CPU](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_cpu_rate_control_information), [ограничение памяти](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information), [показатели нагрузки дисков](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw). `SetIoRateControlInformationJobObject` [не поддерживается в современных Windows](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-setioratecontrolinformationjobobject), поэтому он не используется.
