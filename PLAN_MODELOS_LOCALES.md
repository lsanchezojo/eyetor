# Plan De Trabajo: Calidad Con Modelos Locales Pequenos

## Objetivo

Mejorar Eyetor para obtener la maxima calidad posible usando modelos pequenos locales via `llama.cpp` u Ollama, reduciendo carga cognitiva del prompt, endureciendo el uso de herramientas y haciendo mas robustas las fases auxiliares: routing, KB, sintesis y evaluacion.

## Reglas Para El Agente

- No modificar configuracion relacionada con `ssl_verify`.
- Mantener compatibilidad con APIs OpenAI-compatible.
- Priorizar `llamacpp` y `ollama` como backends principales.
- No hacer grandes refactors si un cambio local resuelve el problema.
- Anadir tests para cada bug o comportamiento fragil corregido.
- Ejecutar `pytest -q` al final.

## Fase 1: Prompt Thinning Para SLMs

### Problema

Actualmente se inyectan instrucciones completas de todas las skills en el system prompt desde `SkillRegistry.build_skills_context`, lo que degrada mucho a modelos pequenos.

### Cambios

- En `src/eyetor/skills/registry.py`, crear un metodo tipo `build_skills_summary_context()` que incluya solo nombre, descripcion y scripts publicos.
- Mantener `build_skills_context()` para comandos manuales como `eyetor skills info`, pero no usarlo por defecto en el system prompt.
- En `src/eyetor/cli.py`, sustituir la inyeccion completa de skills por el resumen.
- Anadir un mecanismo opcional para que rutas/toolsets concretos puedan anadir instrucciones ampliadas solo de las skills relevantes.

### Criterios De Aceptacion

- El system prompt inicial no contiene cuerpos completos de `SKILL.md`.
- `/skills` y `/tools` siguen funcionando.
- Los tools de skills siguen registrandose igual.

## Fase 2: Parsing Robusto De Tool Calls

### Problema

`_parse_completion_response` asume que cada tool call trae `id` y que `arguments` es string. Algunos modelos o servidores locales devuelven tool calls incompletas o argumentos como objeto.

### Cambios

- En `src/eyetor/providers/openrouter.py`, robustecer `_parse_completion_response`.
- Si falta `tc["id"]`, generar uno estable con `uuid.uuid4().hex[:24]`.
- Si `arguments` es dict/list, serializar con `json.dumps(..., ensure_ascii=False)`.
- Si falta `function.name`, ignorar esa tool call y registrar warning.
- Reutilizar ese parser desde `llamacpp.py`, `ollama.py`, `openrouter.py` y revisar `gemini.py` para alinear comportamiento.

### Tests

- Respuesta con tool call sin `id`.
- Respuesta con `arguments` como dict.
- Respuesta con tool call malformada.
- Respuesta normal sigue parseando igual.

## Fase 3: Config Local-First

### Problema

`.env.example` declara `LLAMACPP_API_KEY`, pero `config/default.yaml` no lo usa. Ademas el fallback recomendado debe estar orientado a local.

### Cambios

- En `config/default.yaml`, anadir `api_key: ${LLAMACPP_API_KEY}` bajo `providers.llamacpp`.
- Revisar si `providers.ollama` deberia aceptar `api_key` opcional para proxies locales.
- Cambiar la cadena recomendada en docs/config comentada a modo local-first: `llamacpp`, luego `ollama`.
- Mantener proveedores cloud configurables, pero documentarlos como fallback opcional.

### Criterios De Aceptacion

- Config sigue cargando si `LLAMACPP_API_KEY` esta vacio.
- README refleja perfil local-first.
- No se toca nada relacionado con `ssl_verify`.

## Fase 4: Opciones Avanzadas Por Proveedor

### Problema

Falta control fino de calidad para modelos pequenos: `top_p`, `repeat_penalty`, `num_ctx`, `num_predict`, `stop`, `options`, `extra_body`.

### Cambios

- Extender `ProviderConfig` en `src/eyetor/config.py` con:
  - `max_tokens` o `num_predict`
  - `top_p`
  - `top_k`
  - `repeat_penalty`
  - `stop`
  - `extra_body: dict[str, Any] = {}`
  - `options: dict[str, Any] = {}` para Ollama
- En `BaseProvider._build_payload`, incluir campos comunes si estan definidos.
- En `LlamaCppProvider`, mezclar `extra_body` con payload final sin pisar `messages`, `model`, `tools`.
- En `OllamaProvider`, enviar `options` cuando aplique.

### Criterios De Aceptacion

- Backwards compatible: configs existentes siguen funcionando.
- Los campos extra no se envian si no estan definidos.
- Tests unitarios de payload para llama.cpp y Ollama.

## Fase 5: Perfiles De Tarea

### Problema

La misma temperatura, thinking y prompt se usan en tareas muy distintas. Un SLM necesita modos estrechos.

### Cambios

- Anadir config tipo:
  - `profiles.chat`
  - `profiles.tool_use`
  - `profiles.classifier`
  - `profiles.kb_research`
  - `profiles.synthesis`
- Cada perfil puede definir `temperature`, `thinking`, `max_tool_calls`, `max_wall_seconds`, `extra_body/options`.
- Aplicar perfiles en:
  - routing classifier
  - chat sin tools
  - generic tool loop
  - KB research
  - final synthesis
  - compaction

### Criterios De Aceptacion

- Si no hay perfiles, comportamiento actual se mantiene.
- Las llamadas auxiliares usan `thinking=False` por defecto.
- Las llamadas de investigacion pueden usar thinking si esta configurado.

## Fase 6: Salida Estructurada Para Routing Y Evaluacion

### Problema

El router y evaluator piden JSON por prompt, pero modelos pequenos devuelven texto alrededor.

### Cambios

- Crear helper comun `extract_json_object(text: str)`.
- Usarlo en:
  - `src/eyetor/workflows/router.py`
  - `src/eyetor/workflows/evaluator.py`
  - `src/eyetor/workflows/orchestrator.py`
- Si el proveedor soporta `response_format` o grammar via `extra_body`, permitir activarlo para classifier/evaluator.
- Mejorar fallback: si el JSON es invalido, elegir ruta por scoring lexico contra nombres/descripciones, no solo por aparicion del nombre.

### Tests

- JSON valido.
- JSON dentro de Markdown.
- Texto con JSON embebido.
- Texto sin JSON pero con nombre de ruta.
- Texto sin ruta reconocible.

## Fase 7: KB/RAG De Alta Calidad

### Problema

BM25-only es ligero pero limita calidad semantica.

### Cambios

- Mantener BM25 por defecto si el usuario quiere bajo consumo.
- Anadir perfil documentado "quality KB" con embeddings locales pequenos.
- Mejorar `send_kb_query`:
  - Si `kb_search` devuelve pocos resultados, probar segunda query reformulada.
  - Si hay resultados, leer automaticamente el top 1 o top 2 si no se ha leido.
  - En sintesis, exigir "no inventes si las notas no bastan".
- Anadir metadatos mas claros al scratchpad: `doc_id`, `path`, `heading`, `score`.

### Tests

- KB sin tools disponibles.
- `kb_search` sin resultados.
- `kb_read` con seccion inexistente.
- Sintesis recibe `doc_id`, `path` y `heading`.

## Fase 8: Reducir Descripciones De Tools

### Problema

Algunas descripciones de tools, especialmente scheduler, son largas y consumen contexto.

### Cambios

- Crear descripciones cortas para schemas enviados al modelo.
- Mover reglas largas a prompts de ruta o documentacion interna.
- Para `schedule_task`, mantener validacion defensiva en codigo, no solo en prompt.
- Revisar descriptions en `cli.py` y `chat/session.py`.

### Criterios De Aceptacion

- Tool schemas siguen siendo claros.
- Menos tokens enviados por turno.
- No se elimina validacion de medianoche/hora ausente.

## Fase 9: Metricas De Calidad

### Cambios

- Registrar por turno:
  - ruta elegida
  - confidence
  - numero de tool calls
  - loops detectados
  - fallback provider usado
  - respuesta vacia recuperada
  - forced final answer
- Guardar en logs o SQLite existente si encaja con `tracking`.

### Criterios De Aceptacion

- No rompe tracking actual.
- Permite diagnosticar por que una respuesta fue mala.

## Fase 10: Suite De Evaluacion Local

### Cambios

Crear `tests/evals/` o `scripts/eval_local.py` con prompts representativos:

- charla sin tools
- pregunta KB
- busqueda web vacia
- filesystem/shell simple
- scheduler
- tool call textual tipo `<tool_call>`
- JSON invalido del classifier
- contexto largo con compaction
- imagen/vision si hay mock

### Criterios De Aceptacion

- Puede ejecutarse sin servidores reales usando providers fake.
- Produce reporte simple: pass/fail, ruta esperada, tools esperadas, salida no vacia.

## Orden Recomendado

1. Parsing robusto de tool calls.
2. Prompt thinning.
3. Tests de providers/router/session.
4. Config local-first.
5. Opciones avanzadas llama.cpp/Ollama.
6. Perfiles por tarea.
7. KB quality mode.
8. Reduccion de descripciones de tools.
9. Metricas.
10. Eval suite.

## Verificacion Final

Ejecutar:

```powershell
pytest -q
eyetor providers test llamacpp
eyetor providers test ollama
eyetor run "hola, responde breve"
eyetor run "lista las tools disponibles"
eyetor kb search "consulta de prueba"
```

## Entregable Esperado

PR con cambios pequenos, tests nuevos y README actualizado con un perfil recomendado para modelos locales pequenos.
