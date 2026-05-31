# vLLM load tests

Herramienta para hacer pruebas de carga contra un servidor **vLLM** con API
compatible OpenAI (típicamente desplegado en RunPod).

Mide, por nivel de concurrencia:

- **Latencia end-to-end** (avg / p50 / p90 / p99 / max)
- **TTFT** (Time To First Token) — clave para percepción de respuesta
- **Inter-token latency** — fluidez del streaming
- **Throughput** — `req/s` y `output-tokens/s`
- Errores (con muestras de los primeros distintos)

Soporta dos endpoints (`--mode`):

- `chat` (por defecto) → `POST /v1/chat/completions`
- `responses` → `POST /v1/responses` (Responses API)

Por defecto usa *streaming* (`stream: true`) para poder medir TTFT. Con
`--no-stream` se piden respuestas completas de una vez (como muchos clientes
reales): en ese modo no hay TTFT ni inter-token latency, pero sí latencia e2e
y throughput.

## Tipos de completion (`--completion-type`)

Para cada tipo, el script usa un **system prompt coherente** que fija la salida:
le dice al modelo que, *independientemente* de lo que diga el usuario (aunque
sea ininteligible, vacío o contradictorio), produzca **siempre la misma salida**.
Esto mantiene la salida acotada y determinista, así puedes aislar el coste de
*prefill/decode* sin que el contenido cambie de una petición a otra.

| Tipo | Qué fuerza | Salida fija |
|------|-----------|-------------|
| `text` (def.) | Texto libre | `OK` |
| `json` | JSON estructurado (`response_format` json_schema / guided decoding) | `{"status":"ok","code":200}` |
| `tool` | **Una** función forzada (`tool_choice` con nombre) | llamada a `report_status(status="ok", code=200)` |
| `tools` | Varias tools, `tool_choice=required` (el modelo elige) | llamada a `report_status` |

El `json` y los `tool/tools` activan *guided decoding* en vLLM (gramática/FSM),
cuyo overhead es interesante de medir frente a `text`. Puedes sustituir el
system prompt con `--system-prompt "..."`.

## Prompts muy grandes (p.ej. 200k tokens)

El contenido se separa en dos partes:

- **system prompt**: la instrucción coherente y pequeña (ver arriba).
- **user content**: el relleno gigante (`--prompt-tokens`) que el system manda ignorar.

Así puedes mandar 200k tokens manteniendo la coherencia (la salida sigue siendo
fija y corta). El relleno se genera en O(n) (se tokeniza una vez y se trocea).
El comportamiento frente al *prefix caching* de vLLM es configurable — ver la
sección siguiente.

## Prefix caching y warmup (`--cache`, `--salt`, `--warmup`)

vLLM cachea el *prefill* de prefijos de prompt ya vistos. Esto cambia
radicalmente el TTFT en contextos grandes, así que el script lo parametriza:

- `--cache off` (por defecto): cada petición lleva un **prefijo único** al
  inicio (`Variant {n}`), por lo que un primer token distinto invalida todo el
  prefijo cacheado y **cada petición paga un prefill en frío**. Es el modo para
  medir honestamente el coste de prefill.
- `--cache on`: **todas** las peticiones comparten un prompt **idéntico**, así
  que a partir de la primera son *prefix-cache hits* (camino caliente). Útil
  para medir cuánto ahorra la caché.

vLLM mantiene el prefix cache **entre ejecuciones**, lo que puede contaminar
medidas (una segunda corrida con los mismos prompts aparece artificialmente
rápida). `--salt <str>` antepone una cadena a todos los prefijos:

- pasa un valor fresco por corrida (p.ej. `--salt run3-`) para forzar un
  **arranque en frío** garantizado, o
- mantenlo estable para **reutilizar** una caché calentada en una corrida previa.

Con `--cache on`, `--warmup N` envía `N` peticiones **secuenciales** antes de la
tanda cronometrada (sus resultados se descartan) para **poblar la caché**; así
la medida concurrente refleja el camino caliente y no paga el prefill en frío
inicial de una sola petición.

> Con `--cache off` el warmup solo calienta `prompt[0]` (no los prefijos únicos
> de cada petición), así que tiene poco efecto; el script lo avisa.

> ⚠️ Para 200k tokens el server vLLM debe haberse arrancado con
> `--max-model-len` suficiente (≥ prompt + max-tokens) y el modelo debe soportar
> ese contexto. Si no, verás errores 400 de longitud.
>
> El conteo de tokens es **aproximado**: `tiktoken` (cl100k) no es el tokenizer
> real de tu modelo. El script imprime el conteo real estimado al arrancar.

## Métricas del servidor (`--metrics`)

Con `--metrics` el script raspa el endpoint Prometheus de vLLM
(`<base-url>/metrics`) **antes y después** de cada nivel (deltas de contadores e
histogramas) y **muestrea los gauges durante** la tanda (cada
`--metrics-interval`, 0.5s por defecto). Da la **visión del servidor**, que
complementa las métricas de cliente:

- **Prefix cache hit rate** (en tokens) y **prompt tokens servidos de caché** →
  confirma numéricamente el efecto de `--cache on/off`.
- **KV computed/req**: tokens KV *nuevos* (no cacheados) por petición — la prueba
  directa del cacheo (frío ≈ prompt completo; caliente ≈ 0).
- **Latency split servidor**: cola (`WAITING`) / prefill / decode / TTFT / e2e,
  promediados desde los histogramas. Explica de qué se compone el TTFT.
- **Running / Waiting / KV cache usage** (picos y media) → cuánto encola y
  cuánta KV cache consume el pod bajo carga (en reposo son 0, por eso se
  muestrean en vivo).
- **Preemptions** → recomputación por presión de KV.
- **Finished reasons** (`stop` / `length`) → si las peticiones acaban por tope
  de tokens o de forma natural.

> El delta de `/metrics` captura **todo** el tráfico al servidor en esa ventana,
> no solo el de esta herramienta. En un pod dedicado de pruebas (un único
> cliente) es exacto; si hay más clientes, los contadores estarán inflados.
>
> El **TTFT servidor** suele ser menor que el **TTFT cliente**: la diferencia es
> la red + el parsing del streaming en el cliente.

```bash
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net --metrics \
  --cache on --warmup 1 \
  --concurrency 4 --num-requests 16 --prompt-tokens 8000 --max-tokens 64
```

## Requisitos

Las dependencias (`httpx`, `tiktoken`) ya están en el `.venv` del proyecto.
Ejecuta el script con ese intérprete:

```bash
.venv/bin/python dev/vllm_load_tests/load_test.py --help
```

> `tiktoken` se usa solo para **contar** tokens al construir prompts de tamaño
> controlado. Si no estuviera, hay un fallback heurístico (~0.75 palabras/token).

## Endpoint en RunPod

vLLM escucha por defecto en el puerto `8000`. En RunPod la URL suele ser:

```
https://<POD_ID>-8000.proxy.runpod.net
```

Si arrancaste vLLM con `--api-key`, pásalo con `--api-key`.

## Ejemplos

**Concurrente** — 50 peticiones, 10 en paralelo, prompt ~512 tokens, 128 de salida:

```bash
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --concurrency 10 --num-requests 50 \
  --prompt-tokens 512 --max-tokens 128
```

**Secuencial** — baseline de una petición cada vez (límite inferior de latencia):

```bash
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --sequential --num-requests 20 \
  --prompt-tokens 256 --max-tokens 128
```

**Barrido (sweep)** — varios niveles de concurrencia de una pasada + tabla resumen:

```bash
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --concurrency 1,2,4,8,16,32 --num-requests 64 \
  --prompt-tokens 512 --max-tokens 128 \
  --json-out resultados.json
```

**Responses API** (`/v1/responses` en vez de chat completions):

```bash
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --mode responses --concurrency 8 --num-requests 32 \
  --prompt-tokens 512 --max-tokens 128
```

**Sin streaming** (respuesta completa de una vez; mide latencia total, no TTFT):

```bash
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --no-stream --concurrency 8 --num-requests 32 \
  --prompt-tokens 512 --max-tokens 128
```

**Prompt enorme (~200k tokens) con salida fija coherente:**

```bash
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --sequential --num-requests 5 \
  --prompt-tokens 200000 --max-tokens 16
```

**Camino caliente vs frío (efecto del prefix cache):**

```bash
# Frío: cada petición un prefill nuevo (salt fresco para ignorar caché previa)
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --cache off --salt cold-$(date +%s)- \
  --concurrency 4 --num-requests 16 --prompt-tokens 8000 --max-tokens 64

# Caliente: prompt compartido + warmup que puebla la caché antes del cronómetro
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --cache on --warmup 1 \
  --concurrency 4 --num-requests 16 --prompt-tokens 8000 --max-tokens 64
```

**Salida JSON estructurada (guided decoding):**

```bash
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --completion-type json --concurrency 8 --num-requests 32 \
  --prompt-tokens 512 --max-tokens 64
```

**Tool forzada / varias tools:**

```bash
# Una función forzada
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --completion-type tool --concurrency 8 --num-requests 32 \
  --prompt-tokens 512 --max-tokens 64

# Varias tools, el modelo debe llamar a una (tool_choice=required)
.venv/bin/python dev/vllm_load_tests/load_test.py \
  --base-url https://<POD_ID>-8000.proxy.runpod.net \
  --completion-type tools --concurrency 8 --num-requests 32 \
  --prompt-tokens 512 --max-tokens 64
```

## Opciones

| Flag | Por defecto | Descripción |
|------|-------------|-------------|
| `--base-url` | (obligatorio) | URL base del servidor vLLM |
| `--model` | autodetect | Id del modelo; si se omite se consulta `/v1/models` |
| `--api-key` | — | Bearer token si el server lo exige |
| `--mode` | `chat` | `chat` → `/v1/chat/completions`, `responses` → `/v1/responses` |
| `--completion-type` | `text` | `text` / `json` / `tool` / `tools` (ver sección arriba) |
| `--system-prompt` | built-in | Sustituye el system prompt de salida fija |
| `--no-stream` | off | Desactiva streaming (respuesta completa de una vez; sin TTFT) |
| `--num-requests` | 50 | Nº total de peticiones (por nivel de concurrencia) |
| `--concurrency` | 10 | Nivel, o lista `1,2,4,8` para barrido |
| `--sequential` | off | Una petición cada vez (concurrencia=1) |
| `--cache` | `off` | `off` = prefijo único por petición (prefill en frío); `on` = prompt compartido (prefix-cache hits) |
| `--salt` | — | Cadena antepuesta a los prefijos para invalidar el prefix cache entre corridas |
| `--warmup` | 0 | Nº de peticiones de warmup (descartadas) antes del cronómetro; pobla la caché con `--cache on` |
| `--prompt-tokens` | 512 | Tamaño aproximado del prompt en tokens |
| `--max-tokens` | 128 | Tokens máximos de salida por petición |
| `--temperature` | 0.0 | Temperatura (0 = determinista) |
| `--timeout` | 300 | Timeout por petición (s) |
| `--metrics` | off | Raspa `/metrics` de vLLM (visión servidor: caché, split prefill/decode, KV%, preempciones) |
| `--metrics-url` | `<base>/metrics` | Endpoint de métricas alternativo |
| `--metrics-interval` | 0.5 | Segundos entre muestras de los gauges durante la carga |
| `--json-out` | — | Ruta para volcar el resumen en JSON (incluye `server` si `--metrics`) |

## Cómo interpretar los resultados

- **TTFT** sube con la concurrencia → la cola de *prefill* se satura.
- **Inter-token latency** sube con la concurrencia → el batch de *decode* compite por GPU.
- **output-tokens/s** agregado debería **crecer** con la concurrencia hasta saturar
  la GPU; cuando deja de crecer (o las latencias se disparan) has encontrado el
  punto de saturación del pod.
- El modo **secuencial** te da el **mejor caso** de latencia (sin contención),
  útil como referencia frente a los niveles concurrentes.

- **Prefix cache** (`--cache on` vs `off`): comparar ambos te da el ahorro real
  del *prefill* cacheado. En frío el TTFT está dominado por el prefill; en
  caliente el TTFT cae a casi el de un prompt corto y lo que queda es el decode.

> Nota: cada nivel de concurrencia reutiliza el **mismo conjunto de prompts** para
> que la comparación sea justa. Por defecto (`--cache off`) los prompts llevan un
> prefijo único por petición para no falsear los resultados con el *prefix
> caching* de vLLM; usa `--cache on` (opcionalmente con `--warmup`) para medir
> explícitamente el camino caliente.
