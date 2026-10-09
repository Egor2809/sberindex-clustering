# DMoN: эталонная модель совместной кластеризации

Добавлена реализация DMoN на PyTorch для сравнения на тех же признаках и дорожном графе. Это перенос метода по статье с архитектурой из официального кода, а не воспроизведение опубликованных чисел или побитовое совпадение с TensorFlow. Результат на данных проекта нужно оценивать отдельно; преимущество перед другими моделями заранее не предполагается.

## Источники и версия

Anton Tsitsulin, John Palowitch, Bryan Perozzi, Emmanuel Müller. **Graph Clustering with Graph Neural Networks**, JMLR 24(127):1–21, 2023. [Страница статьи](https://www.jmlr.org/papers/v24/20-998.html), [PDF](https://www.jmlr.org/papers/volume24/20-998/20-998.pdf). Модульность определена формулами (1–2) на с. 5, архитектура и функция потерь DMoN описаны формулами (3–5) на с. 6.

Официальный код Google Research закреплён на commit `5b09c22d73a9d35eb6c5d2a99b95677a45053466`:

- [dmon.py](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/graph_embedding/dmon/dmon.py): softmax, модульность и collapse loss.
- [gcn.py](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/graph_embedding/dmon/gcn.py): sparse GCN, обучаемый поканальный skip и SELU.
- [train.py](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/graph_embedding/dmon/train.py): скрытый слой 64, Adam, learning rate 0.001, collapse weight 1, dropout 0 по умолчанию; официальный бюджет 1000 эпох.
- [utils.py](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/graph_embedding/dmon/utils.py): нормировка графа.

Код Google Research распространяется по Apache License 2.0. Модуль `sbercluster/dmon.py` содержит изменённую реализацию на PyTorch с сохранённым уведомлением об авторстве. Текст лицензии приведён ниже. Статья имеет отдельную лицензию CC-BY 4.0.

## Точная функция потерь

Пусть `A` обозначает симметричная неотрицательная матрица смежности без петель, `d=A1`, `v=sum(d)=2m`, а `S` обозначает матрицу мягких назначений `n×K`, строки которой суммируются в 1. Оптимизируется ровно формула (5):

```text
Q_soft = [sum(S * (A @ S)) - sum((d @ S)^2) / v] / v
collapse = sqrt(K) / n * ||sum_rows(S)||_2 - 1
loss = -Q_soft + collapse
```

Нет дополнительных потерь реконструкции, энтропии, ортогональности, расстояний до центров или внешних откликов. Жёсткие метки берутся как `argmax(S, axis=1)`. Мягкая модульность в функции потерь и модульность итогового жёсткого разбиения дают разные числа.

Для взвешенного дорожного графа степени заменяются суммами весов, а `v` удвоенным суммарным весом неориентированных рёбер. Это явно обозначенное расширение постановки статьи, где вводится бинарная смежность. Веса не переводятся в бинарные. Изоляты остаются в модели: строка нормированного графа у них нулевая, признаки проходят через skip; их назначения учитываются в collapse loss.

## Архитектура и отличия

```text
P = X @ W                         # n×64
H = SELU(D^-1/2 A D^-1/2 @ P + P * skip + bias)
S = softmax(H @ W_out + bias_out) # n×K
```

Один скрытый слой, 64 канала; линейный слой назначений имеет ортогональную инициализацию, выходной bias нулевой. GCN kernel использует Xavier uniform; skip использует uniform в пределах `±sqrt(3/64)`; GCN bias установлен в ноль явно. Dropout равен нулю, как default в официальном `train.py`; пример Cora в README использует 0.5 и здесь не воспроизводится. Adam использует `lr=0.001`, `eps=1e-7`, без weight decay. CPU float32; исходные признаки не масштабируются повторно внутри модели.

Существенные расхождения с закреплённым кодом указаны открыто:

1. В `dmon.py` переменная `number_of_edges` равна сумме степеней, но нормализатор и спектральная потеря дополнительно делятся на 2. При обычной симметричной смежности это даёт знаменатель `4m`, в отличие от `2m` в формулах (1), (2), (5). Здесь применена формула статьи; тест отдельно проверяет её на прямой плотной матрице модульности и ручном примере. Старый `layers/modularity.py` содержит такую же нормировку и дополнительный orthogonality loss; этот вариант не используется.
2. `utils.normalize_graph` по умолчанию добавляет единичные петли. На с. 6 статьи сказано об их удалении в пользу обучаемого skip. Здесь петли не добавляются; входные петли отклоняются, чтобы не менять граф сравнения молча.
3. Skip реализован поканальным вектором после общей проекции, как в официальном `gcn.py`. Запись формулы (3) в статье допускает более общий skip; отдельная полная skip-матрица здесь не вводится.
4. GCN bias нулевой при старте; в исходном `add_variable` его инициализация оставлена Keras. Генераторы случайных чисел, детали Adam и арифметика PyTorch отличаются от TensorFlow. Совпадение seed между библиотеками не означает совпадения параметров.
5. Сохраняется состояние с минимальной наблюдённой **обучающей** функцией потерь, включая эпоху 0; официальный цикл возвращает последнее. Никакие внешние метрики не используются для выбора checkpoint. При равенстве сохраняется более раннее состояние. Первая серия имела фиксированный бюджет 1 000 эпох; продолжение 25 сентября использует описанное ниже эмпирическое правило плато.

Это DMoN по функции потерь статьи с явно описанной архитектурой, а не новый метод. Полная матрица `n×n` не создаётся и не обучается; распространение использует sparse matrix multiplication. Память модели растёт как `O(nnz(A)+n·(d+h+K))`, стоимость прохода включает `O(nnz(A)·(h+K)+n·d·h+n·h·K)`.

## Использование и проверка

PyTorch остаётся отдельной необязательной зависимостью: `requirements-dmon.txt`. Основной пакет требований не изменён. Для CPU-окружения можно установить её через официальный индекс CPU wheels:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-dmon.txt
python -m unittest discover -s tests -p test_dmon.py -v
```

Вызов из собственного контролируемого сценария:

```python
from sbercluster.dmon import fit_dmon
labels, info, assignments = fit_dmon(
    x, adjacency, n_clusters=4, seed=1729, epochs=200,
    hidden_dim=64, learning_rate=0.001,
    return_assignments=True, output_dir="runs/dmon-example",
)
```

`x` и `adjacency` должны иметь одинаковый порядок территорий. Проверка отклоняет направленные, отрицательные, пустые графы и графы с петлями. Симметрия принимается с абсолютным допуском `1e-6`; принятая погрешность устраняется усреднением транспонированных значений. Преобразование входа в float32 описано в метаданных.

Сохраняются `trace.json` (эпохи 0..budget), `info.json`, `best_model.npz` (параметры без pickle), `assignments.npy` и `labels.npy`. Checkpoint пригоден для восстановления назначений; состояние Adam для возобновления обучения не сохраняется. Число занятых жёстких кластеров и их размеры всегда сообщаются: `argmax` может дать меньше K групп, их искусственное заполнение не производится.

Тесты на маленьких графах сравнивают loss и градиенты по назначениям, признакам, параметрам и весам рёбер с плотной аналитической формулой; проверяют изоляты, ручной пример модульности, детерминированное повторение в одной среде и точное восстановление checkpoint. Если Torch отсутствует, тесты пропускаются с явной причиной. Эти проверки не подтверждают качество кластеризации реальных территорий.

Для первого сравнения разумен небольшой фиксированный пилот на одинаковых входах и K, затем единый бюджет для всей сравниваемой серии по обучающим трассам пилота. При 2 016 вершинах, 18 673 неориентированных рёбрах и 64 каналах основной sparse GCN проход содержит около 2,39 млн произведений ненулевого веса на канал. Это оценка порядка вычислений, не измерение времени или обещание сходимости. Бюджет 100–200 эпох является пилотом, а не воспроизведением официальных 1000 эпох. Позднее улучшение loss требует явно продлённого бюджета; выход по бюджету не называется сходимостью. Результаты разных seed нужны для оценки устойчивости. Не следует выбирать seed по внешнему отклику, силуэту или исходам 2024 года.

## Apache License 2.0

```text

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```


## Продолжение обучения до плато, 25 сентября

Серия `configs/dmon_plateau.json` сохраняет модель, входы, learning rate и шесть сочетаний K/seed. Старые файлы содержали только веса для inference, поэтому обучение повторяет старт с того же seed, восстанавливая историю обновлений Adam. Метки эпохи 1 000 сохранены отдельно и сопоставляются со старой серией; различие сборок PyTorch допускает небольшую численную разницу loss.

После эпохи 3 000 каждые 500 эпох сравниваются лучший обучающий loss и hard labels лучшего checkpoint с предыдущей проверкой. Остановка требует трёх последовательных окон с улучшением loss не более 0,0001 и изменением не более 0,1% меток. Максимум составляет 50 000 эпох; достижение этого предела записывается отдельно от плато. Выбирается checkpoint с минимальным loss, включая инициализацию; внешний показатель и силуэт не участвуют в остановке. Это эмпирический критерий, а не доказательство стационарности или глобального оптимума.
