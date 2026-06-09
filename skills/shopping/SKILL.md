---
name: shopping
description: Gestiona compras desde tickets fotografiados. Extrae productos y precios, mantiene la lista de la compra, consulta historial y planifica la ruta más barata o con menos visitas entre tiendas.
license: MIT
compatibility: Python 3.11+. Solo stdlib (sqlite3, difflib, urllib, base64). Necesita un proveedor de visión configurado en runtime.json para reconfirmaciones de ticket.
metadata:
  author: eyetor
  version: "0.1"
timeout: 180
commands:
  - name: recibo
    description: Registra el último ticket fotografiado.
    action: prompt
    prompt: |
      El usuario ha enviado una foto de ticket (o quiere registrar uno). Si está disponible,
      usa primero la herramienta estructurada `shopping_receipt_add` para registrarlo. Si no
      está disponible, usa la skill 'shopping' (script receipt.py). Para la fecha:
      usa primero la fecha visible en la imagen; si falta, busca una fecha completa en el
      caption/mensaje del usuario; si tampoco aparece ahí, pregunta la fecha al usuario antes
      de registrar. Si el análisis visual previo no es suficiente por otros campos (faltan
      precios, no cuadra total o falta tienda), llama
      a `receipt.py reconfirm --image-path <ruta>` sobre la imagen indicada en la línea
      "Imagen guardada en: ..." del contexto. Tras registrar, si hay coincidencias con la
      lista de la compra, pide confirmación al usuario antes de eliminarlas. {args}
  - name: lista
    description: Lista de la compra. Añadir, quitar, mostrar, vaciar.
    action: prompt
    prompt: |
      Gestiona la lista de la compra del usuario usando la skill 'shopping' (script list.py).
      Operaciones disponibles: add, remove, show, clear. Cuando añadas un ítem, vincula el
      producto a su canonical si el script lo sugiere con score ≥ 0.85 (auto-link). Si está
      entre 0.70 y 0.85, pregunta al usuario antes de enlazar. Petición: {args}
  - name: precios
    description: Historial de precios o producto más barato.
    action: prompt
    prompt: |
      Consulta precios con la skill 'shopping' (script price.py). Si el usuario pide "dónde
      está más barato X", usa `price.py cheapest --product "..."`. Si pide histórico, usa
      `price.py history --product "..."` opcionalmente con --store o --days. Petición: {args}
  - name: optimizar
    description: Plan óptimo para la lista (más barato o menos visitas).
    action: prompt
    prompt: |
      Calcula con la skill 'shopping' (script optimize.py plan) la mejor estrategia para
      comprar la lista actual. Modo "cheapest": precio mínimo por producto entre tiendas.
      Modo "fewer-stores": minimiza el número de tiendas visitadas (greedy set-cover,
      opcional --max-stores N). Petición: {args}
---

# shopping — compras, lista y optimización

Skill para registrar tickets de supermercado a partir de fotos, mantener una lista de la
compra y responder preguntas como "¿dónde es más barato X?" o "¿qué ruta de tiendas me
ahorra visitas?".

Toda la persistencia vive en una sola base de datos SQLite global:
`~/.eyetor/shopping.db`. Cada **unidad** comprada se almacena como una fila independiente
(si en un ticket aparecen 3 leches, son 3 filas).

## Cuándo usar esta skill

- El usuario envía una foto de un ticket o menciona "ticket", "recibo", "compra".
- El usuario habla de su lista de la compra (añadir, quitar, mostrar, vaciar).
- Pregunta dónde es más barato un producto, o cómo organizar la compra entre tiendas.
- Pregunta por el historial de precios de un producto.

## Flujo de un ticket por foto

1. El usuario manda foto a Telegram. El canal guarda la imagen en
   `~/.eyetor/images/{chat_id}_{ts}.jpg` y añade al contexto una línea
   `Imagen guardada en: <path>` junto con la descripción del modelo de visión y,
   cuando existe, el caption del usuario.

2. Extrae del texto descriptivo: `store`, `date` (formato `YYYY-MM-DD`), `items`
   (lista de `{name, price}` con **precio unitario**), `total` si aparece.
   Si un producto aparece N veces en el ticket, **emite N entradas idénticas** en
   `--items` (una por unidad). Si no hay fecha visible en la imagen, busca una fecha
   completa en el caption/mensaje del usuario y úsala. Si tampoco hay fecha en el
   caption, pregunta al usuario antes de registrar.

3. Si está disponible, llama a `shopping_receipt_add` con parámetros estructurados:

       {
         "store": "Mercadona",
         "date": "2026-05-11",
         "items": [
           {"name": "Leche entera 1L", "price": 1.05},
           {"name": "Leche entera 1L", "price": 1.05},
           {"name": "Pan", "price": 0.85}
         ],
         "total": 2.95,
         "image_path": "/home/haziel/.eyetor/images/123_1715000000.jpg"
       }

   Si esa herramienta no está disponible, llama:

       receipt.py add --store "Mercadona" --date 2026-05-11 \
           --items '[{"name":"Leche entera 1L","price":1.05},
                     {"name":"Leche entera 1L","price":1.05},
                     {"name":"Pan","price":0.85}]' \
           --total 2.95 \
           --image-path /home/haziel/.eyetor/images/123_1715000000.jpg

4. Posibles respuestas del script:

   - `{"ok":true, "receipt_id":N, "inserted":3, "reconcile":[...]}` → todo bien.
   - `{"ok":true, "needs_reconfirm":true, "reason":"..."}` → faltan precios o el
     total no cuadra. **No se ha insertado nada**.

5. Si necesita reconfirmación, ejecuta:

       receipt.py reconfirm --image-path /home/haziel/.eyetor/images/123_1715000000.jpg

   El script reprocesa la imagen con el modelo de visión y devuelve
   `{"ok":true, "store":..., "date":..., "total":..., "items":[...]}`. Vuelve al paso 3
   con los datos enriquecidos.

6. Si el `reconcile` no está vacío, redacta un mensaje en lenguaje natural mostrando
   los ítems del ticket que estaban en la lista, y **pide confirmación al usuario**
   antes de borrarlos:

   > "He registrado el ticket de Mercadona (3 productos, 2,95 €).
   > He visto que tenías **Pan** y **Leche entera 1L** en la lista de la compra.
   > ¿Los marco como comprados?"

7. Solo si el usuario confirma, ejecuta:

       list.py remove --ids 3,7

## Lista de la compra

- `list.py add --text "leche entera 1L" [--canonical-id N] [--quantity N]`
- `list.py remove --ids 3,7,12`
- `list.py show [--with-canonical]`
- `list.py clear`

Al añadir, si el script devuelve `suggest_canonical` (score entre 0.70 y 0.85),
pregunta al usuario antes de enlazar con ese canonical.

## Precios y optimización

- `price.py history --product "leche entera" [--store Mercadona] [--days 90]`
- `price.py cheapest --product "leche entera" [--strategy last-known|min-ever|min-last-30d]`
- `optimize.py plan --mode cheapest`
- `optimize.py plan --mode fewer-stores [--max-stores 2]`

`optimize.py` devuelve un plan agrupado por tienda con el coste total y los productos
sin historial en `missing`.

## Mantenimiento del catálogo

- `product.py canonical create --name "Leche entera 1L" [--category lacteos]`
- `product.py canonical list [--search "leche"]`
- `product.py canonical merge --from-id N --into-id M`
- `product.py alias add --alias "le ent 1l" --canonical-id N`
- `product.py alias list --canonical-id N`
- `product.py alias delete --id N`

## Reglas estrictas

- **Nunca inventes precios.** Si un ítem no tiene precio extraído del texto/visión,
  no lo incluyas; deja que el script pida `reconfirm`.
- **Fechas siempre `YYYY-MM-DD`.** Convierte tú "12/05/2026" → "2026-05-12".
  Prioridad: imagen → caption/mensaje → preguntar al usuario. No inventes una fecha.
- **Items duplicados en un ticket: una entrada por unidad** en `--items`. El script
  guarda una fila por unidad para que `MIN/AVG(price)` y las comparaciones funcionen.
- **Reconciliación con la lista: confirmación obligatoria.** No ejecutes
  `list.py remove` sin un "sí" explícito del usuario.
- **Reconfirm sólo si el script lo pide o si el usuario lo solicita.** No
  re-procesa la imagen si `receipt.py add` aceptó el ticket.
