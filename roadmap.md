Ниже — рабочий роадмап именно под задачу “перенести TTS-инференс в Rust/Candle без боли”, а не академический список желаний.

Для **OmniVoice** я бы сразу исходил из того, что это **двухстейджевый пайплайн**:

1. **generator** на базе **Qwen3-0.6B** с **32-шаговым iterative unmasking** и classifier-free guidance, который генерирует **8-codebook audio tokens**;
2. **decoder**: **HiggsAudioV2 RVQ quantizer + DAC acoustic decoder**, который превращает токены в **24 kHz waveform**. Именно поэтому портировать его надо не “целиком”, а по границе stage0/stage1. ([GitHub][1])

## Главный принцип

**Не пиши Rust-реализацию всей модели сразу.**
Делай порт в 4 слоя:

1. **битовая совместимость входов/выходов**
2. **порт stage0**
3. **порт stage1**
4. **сведение в один inference pipeline**

Это почти всегда быстрее и надежнее, чем “сразу собрать полный end-to-end”.

---

# Универсальный роадмап портирования TTS-моделей на Rust + Candle

## Фаза 0. Зафиксировать эталон Python

Сначала делается **golden reference** в Python. Без этого любой Rust-порт превращается в охоту на призраков.

Что сохранить:

* нормализованный входной текст;
* токены текстового токенизатора;
* все служебные id и masks;
* выход **промежуточного уровня**:

  * logits / hidden states / sampled tokens на ключевых шагах;
  * итоговые **audio tokens** stage0;
* выход stage1:

  * декодированные waveform samples;
* конфиг модели, dtype, device, seed, sampling params.

Что нужно получить на выходе:

* 3–5 “канонических” тест-кейсов:

  * короткая фраза;
  * длинная фраза;
  * voice clone;
  * voice design;
  * сложная пунктуация / multilingual.

Зачем: если нет эталона, ты не поймешь, где ошибка — в токенизаторе, позиционках, masking, sampler’е или декодере.

---

## Фаза 1. Разрезать модель на независимые блоки

Для любой TTS-модели сначала делай **architecture map**:

* **text frontend**: normalization, tokenizer, special tokens;
* **acoustic generator**: transformer / diffusion LM / AR / NAR;
* **codec / vocoder / decoder**;
* **sampling loop**;
* **post-processing**.

Для OmniVoice конкретно:

* **frontend**: текст + optional reference/instruction inputs;
* **stage0**: diffusion-LM style generator на Qwen3-0.6B backbone;
* **stage1**: HiggsAudioV2 RVQ + DAC decoder. ([GitHub][1])

Правильная граница портирования:

* сначала сделать Rust-инференс, который получает **готовые audio tokens** и только декодирует их в waveform;
* потом отдельно перенести генерацию токенов;
* потом склеить.

Это самый быстрый путь до первого слышимого результата.

---

## Фаза 2. Решить формат весов и артефактов

Статус на 2026-04-08: **закрыта в текущем workspace**.
Базовый контракт уже зафиксирован через `model/omnivoice.artifacts.json`, manifest-driven loaders в `crates/omnivoice-infer/src/artifacts.rs` и CLI-проверку `omnivoice-cli artifacts validate`.
Повторно перерабатывать Phase 2 не нужно; следующий work unit должен начинаться с `stage1-first` decode parity.

Для Candle лучший базовый путь — хранить веса в **safetensors**. Candle нативно работает с safetensors, а `VarBuilder` умеет загружать веса из mmap’ed safetensors, что удобно и для скорости, и для простоты интеграции. ([Hugging Face][2])

Практически:

* все веса перевести в `.safetensors`;
* tokenizer хранить в `tokenizer.json`, если это возможно;
* все configs сериализовать отдельно в JSON/TOML;
* служебные таблицы, special token maps, speaker metadata — отдельными файлами.

Если какой-то блок плохо переносится в нативный Candle, держи запасной путь:

* экспортировать проблемный сабграф в **ONNX** через Optimum;
* либо временно запускать этот кусок через ONNX Runtime, а остальное держать в Candle. ONNX экспорт в экосистеме HF поддерживается официально. ([Hugging Face][3])

---

## Фаза 3. Поднять минимальный Rust-каркас

Базовый стек:

* `candle-core`
* `candle-nn`
* `candle-transformers` где применимо
* `tokenizers`
* `safetensors`
* `hf-hub` если качаешь модели напрямую с Hub

Candle позиционируется как минималистичный Rust-ML framework с CPU/GPU-бэкендами; у него есть CPU, CUDA, Metal и готовые примеры, в том числе TTS и audio-related examples. ([GitHub][4])

Минимальная структура crate:

* `config.rs`
* `tokenizer.rs`
* `frontend.rs`
* `model/stage0.rs`
* `model/stage1.rs`
* `sampling.rs`
* `audio.rs`
* `pipeline.rs`
* `tests/parity.rs`
* `bin/cli.rs`

---

## Фаза 4. Сначала перенести tokenizer и preprocessing

Это самый недооцененный источник багов.

Лучший путь:

* использовать **тот же tokenizer.json**, если модель совместима с Hugging Face Tokenizers;
* грузить его через Rust crate `tokenizers`, потому что это нативная Rust-реализация того же стека. ([Hugging Face][5])

Что проверить паритетом:

* normalization;
* pre-tokenizer;
* special tokens;
* BOS/EOS/pad/mask;
* language/style/speaker markers;
* prompt templating.

Правило:
**пока токенизация не совпала 1-в-1, к переносу модели не переходить.**

---

## Фаза 5. Перенести stage1 раньше, чем stage0

Для TTS это почти всегда ускоряет разработку.

Почему:

* stage1 проще тестировать;
* на вход ему можно подать audio tokens из Python;
* быстро получаешь WAV на выходе;
* аудио-баги легче локализовать отдельно от генератора.

Для OmniVoice это особенно логично, потому что stage1 отделен как codec/decoder блок: HiggsAudioV2 RVQ + DAC acoustic decoder. ([GitHub][1])

Практика:

1. В Python сохранить `audio_tokens.pt` / `.npy` / `.jsonl`.
2. В Rust загрузить их.
3. Прогнать через stage1.
4. Сравнить waveform:

   * длину;
   * RMS;
   * sample-level MSE / cosine;
   * спектрограмму;
   * субъективное прослушивание.

Если в модели реально используется DAC-компонент, полезно держать под рукой референс по самому DAC/RVQGAN-стеку: Descript Audio Codec — это отдельный известный нейрокодек, применимый как аудио-декодер в подобных пайплайнах. ([GitHub][6])

---

## Фаза 6. Перенести stage0 как “строго детерминированный генератор токенов”

Самая большая ошибка — сразу пытаться повторить “как звучит”.
Сначала нужно повторить **как генерируются audio tokens**.

Для OmniVoice stage0:

* transformer backbone Qwen3-0.6B;
* diffusion language model style;
* 32-step iterative unmasking;
* classifier-free guidance;
* генерация 8 codebooks. ([GitHub][1])

Что переносить по порядку:

1. embeddings
2. positional encoding / rotary / attention masks
3. transformer blocks
4. output heads
5. masking logic
6. iterative loop
7. guidance
8. sampler

Только в этом порядке.

### Критично

Сделай **режим teacher / debug inference**, где:

* sampling отключен;
* вместо случайного выбора берется `argmax`;
* seed зафиксирован;
* dropout и все train-only ветки выключены.

Так ты сначала добьешься совпадения logits / hidden states, а уже потом подключишь стохастику.

---

## Фаза 7. Делать layer-by-layer parity tests

Это самое выгодное вложение времени.

На каждом этапе сверяй:

* shape;
* dtype;
* min/max/mean/std;
* cosine similarity;
* max absolute error;
* несколько конкретных индексов tensor values.

Порог практический:

* embeddings / linear / conv: почти exact, если dtype одинаков;
* fp16/bf16: маленькая погрешность нормальна;
* после softmax/sampling расхождения растут лавинообразно, поэтому сравнивай **до sampling**.

Минимальный тестовый набор:

* `embedding parity`
* `one transformer block parity`
* `full forward parity`
* `one denoise/unmask step parity`
* `audio token parity`
* `waveform parity`

---

## Фаза 8. Не переносить “магические” Python-хелперы вслепую

Обычно тормозят не матоперации, а скрытая логика:

* padding side;
* mask convention;
* ids remapping;
* hidden prompt templates;
* audio token reshaping;
* codebook interleaving;
* chunking/windowing;
* float scaling до PCM.

Поэтому для каждого блока делай отдельный документ:

* **Input contract**
* **Output contract**
* **Shape contract**
* **DType contract**
* **Special cases**

Это экономит дни дебага.

---

## Фаза 9. Сначала correctness, потом optimization

Candle поддерживает CPU и GPU-бэкенды, включая CUDA и Metal. Но оптимизировать имеет смысл только после паритета. ([GitHub][4])

Правильный порядок:

1. CPU + f32 debug build
2. CPU + bf16/f16
3. GPU backend
4. batching / streaming / kv-cache / custom kernels

Иначе ты получишь быстрый, но неверный inference.

---

## Фаза 10. Где реально экономить время

### 1) Не переписывать все руками

Если часть модели уже близка к существующим примерам Candle, переиспользуй паттерны загрузки весов и структуры моделей. У Candle есть готовые примеры, включая **Parler-TTS** и **Encodec**, это хорошие шаблоны для TTS/audio пайплайнов. ([GitHub][7])

### 2) Использовать safetensors + VarBuilder сразу

Это убирает целый класс проблем с загрузкой чекпоинтов. ([GitHub][8])

### 3) Портировать end-to-end по “слышимым milestones”

Порядок milestones:

* Rust декодирует готовые audio tokens в WAV
* Rust генерирует те же audio tokens, что Python
* Rust делает полный TTS
* Rust делает clone/design modes
* Rust делает streaming / batching

---

# Конкретный план именно для OmniVoice

## Этап A. Разведка

За 1 день собрать:

* список всех артефактов модели;
* какие веса уже в safetensors;
* какие tokenizer/config файлы есть;
* какие зависимости Python реально участвуют в inference;
* где живет sampling/masking logic;
* где живет stage1 decoder.

OmniVoice официально позиционируется как zero-shot multilingual TTS на 600+ языков, с auto voice / voice clone / voice design режимами, а vllm-omni уже явно описывает его как two-stage inference pipeline. ([GitHub][9])

## Этап B. Stage1-first

Сделать Rust CLI:

```bash
omnivoice-rs decode-tokens tokens.npy out.wav
```

Цель:

* получить звук из заранее сохраненных Python audio tokens.

## Этап C. Frontend parity

Сделать:

```bash
omnivoice-rs tokenize --text "... " --dump tokens.json
```

И сверить с Python 1-в-1.

## Этап D. Stage0 forward parity

Сделать:

```bash
omnivoice-rs stage0-debug --input sample.json --dump tensors/
```

Сравнить:

* embeddings
* block N outputs
* final logits
* predicted codebooks

## Этап E. Sampling loop parity

Перенести:

* iterative unmasking
* guidance
* mask schedule
* sampler

Сначала deterministic mode, потом normal mode.

## Этап F. End-to-end

Склеить:

```bash
omnivoice-rs synth \
  --text "..." \
  --mode auto|clone|design \
  --ref ref.wav \
  --ref-text "..." \
  --style "female, low pitch, british accent"
```

---

# Что чаще всего ломает порт TTS

1. **Не тот tokenizer / normalization**
2. **Неверный reshape audio tokens**
3. **Перепутанный порядок codebooks**
4. **Неправильные attention masks**
5. **Неверный CFG / guidance scaling**
6. **Off-by-one в positional ids**
7. **Неправильный dtype в critical местах**
8. **Отличия в sampling**
9. **Скрытая логика postprocess audio**
10. **Порт stage0 и stage1 одновременно**

---

# Самый эффективный рабочий шаблон

Я бы делал так:

**Неделя 1**

* golden Python harness
* tokenizer parity
* safetensors conversion
* Rust project skeleton

**Неделя 2**

* stage1 decode from saved tokens
* WAV parity + audio metrics

**Неделя 3**

* stage0 forward parity without sampling
* block-by-block debug

**Неделя 4**

* iterative unmasking + CFG + token generation
* end-to-end inference

**Неделя 5**

* optimization: bf16/f16, CUDA/Metal, batching, latency

---

# Если нужен “универсальный безболезненный” чеклист

Перед портом каждой TTS-модели отвечай на 8 вопросов:

1. Где заканчивается **text frontend**?
2. Где начинается **acoustic token generation**?
3. Есть ли отдельный **codec/vocoder**?
4. Какие артефакты можно сразу использовать в Rust (`tokenizer.json`, `safetensors`)?
5. Что можно протестировать независимо?
6. Где есть стохастика?
7. Какие промежуточные тензоры надо сохранить из Python?
8. Какой минимальный “слышимый milestone” даст confidence?

Если на них есть ответы, порт обычно идет быстро.

---

# Мой практический совет именно для тебя

Для **OmniVoice + Candle** лучший маршрут такой:

**не тащи сразу весь OmniVoice в Rust**;
сделай вначале:

1. **Rust loader + tokenizer**
2. **Rust stage1 decoder от готовых audio tokens**
3. **Rust stage0 deterministic parity**
4. **Rust sampling loop**
5. **full pipeline**

Это даст первый рабочий результат намного раньше и резко снизит риск застрять в дебаге. Основание для такого разбиения у OmniVoice прямое: его inference уже описан как **двухстейджевый** — generator отдельно, decoder отдельно. ([GitHub][1])


[1]: https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/omnivoice "vllm-omni/examples/offline_inference/omnivoice at main · vllm-project/vllm-omni · GitHub"
[2]: https://huggingface.co/docs/transformers/community_integrations/candle?utm_source=chatgpt.com "Candle - Hugging Face"
[3]: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model?utm_source=chatgpt.com "Export a model to ONNX with optimum.exporters.onnx"
[4]: https://github.com/huggingface/candle "GitHub - huggingface/candle: Minimalist ML framework for Rust · GitHub"
[5]: https://huggingface.co/docs/tokenizers/index?utm_source=chatgpt.com "Tokenizers · Hugging Face"
[6]: https://github.com/descriptinc/descript-audio-codec "GitHub - descriptinc/descript-audio-codec: State-of-the-art audio codec with 90x compression factor. Supports 44.1kHz, 24kHz, and 16kHz mono/stereo audio. · GitHub"
[7]: https://github.com/huggingface/candle/tree/main/candle-examples/examples/parler-tts "candle/candle-examples/examples/parler-tts at main · huggingface/candle · GitHub"
[8]: https://github.com/huggingface/candle/blob/main/candle-nn/src/var_builder.rs "candle/candle-nn/src/var_builder.rs at main · huggingface/candle · GitHub"
[9]: https://github.com/k2-fsa/OmniVoice/ "GitHub - k2-fsa/OmniVoice: High-Quality Voice Cloning TTS for 600+ Languages · GitHub"
