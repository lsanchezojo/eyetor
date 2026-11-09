---
name: shopping
description: Gestiona compras desde tickets fotografiados. Extrae productos y precios, mantiene la lista de la compra, consulta historial y planifica la ruta más barata o con menos visitas entre tiendas.
license: MIT
compatibility: Python 3.11+. Solo stdlib (sqlite3, difflib, urllib, base64). Necesita un proveedor de visión configurado en runtime.json para reconfirmaciones de ticket.
metadata:
  author: eyetor
  version: "0.2"
timeout: 180
# Only typed tools (shopping_*): the free-text skill_shopping invited empty
# and malformed calls from small models.
args_tool: false
gating:
  group: shopping
  triggers:
    - compr\w*
    - pag(o|ue|ado|ar|aste|amos)
    - gast\w*
    - lista
    - ticket\w*
    - tique\w*
    - recibo\w*
    - factura\w*
    - total
    - iva
    - supermercado\w*
    - super
    - mercadona|carrefour|lidl|aldi|consum|alcampo|eroski|hipercor|bonpreu|caprabo
    - precio\w*
    - barat\w*
    - car[oa]s?
    - cuesta\w*
    - apunta\w*
    - anad\w*
    - producto\w*
commands:
  - name: recibo
    description: Registra el último ticket fotografiado.
    action: prompt
    prompt: |
      El usuario ha enviado una foto de ticket (o quiere registrar uno). Regístralo con
      `shopping_receipt_add`. Para la fecha: usa primero la fecha visible en la imagen; si
      falta, busca una fecha completa en el caption/mensaje del usuario; si tampoco aparece
      ahí, pregunta la fecha al usuario antes de registrar. Si el análisis visual previo no
      es suficiente por otros campos (faltan precios, no cuadra total o falta tienda), llama
      a `shopping_receipt_reconfirm` con la imagen indicada en la línea "Imagen guardada
      en: ..." del contexto. Tras registrar, si hay coincidencias con la lista de la compra,
      pide confirmación al usuario antes de eliminarlas. {args}
  - name: lista
    description: Lista de la compra. Añadir, quitar, mostrar, vaciar.
    action: prompt
    prompt: |
      Gestiona la lista de la compra del usuario con las herramientas shopping_list_show,
      shopping_list_add, shopping_list_remove y shopping_list_clear. Cuando añadas un ítem,
      vincula el producto a su canonical si la herramienta lo sugiere con score ≥ 0.85
      (auto-link). Si está entre 0.70 y 0.85, pregunta al usuario antes de enlazar.
      Petición: {args}
  - name: precios
    description: Historial de precios o producto más barato.
    action: prompt
    prompt: |
      Consulta precios. Si el usuario pide "dónde está más barato X", usa
      `shopping_price_cheapest`. Si pide histórico, usa `shopping_price_history`,
      opcionalmente con tienda o días. Petición: {args}
  - name: optimizar
    description: Plan óptimo para la lista (más barato o menos visitas).
    action: prompt
    prompt: |
      Calcula con `shopping_plan` la mejor estrategia para comprar la lista actual. Modo
      "cheapest": precio mínimo por producto entre tiendas. Modo "fewer-stores": minimiza
      el número de tiendas visitadas (opcionalmente con max_stores). Petición: {args}
tools:
  - name: shopping_list_show
    description: Show the shopping list (item ids, text, quantity).
    script: list.py
    argv: [show]
    parameters:
      type: object
      properties:
        with_canonical: {type: boolean, description: Include the linked catalog product of each item.}
  - name: shopping_list_add
    description: Add an item to the shopping list. May return a suggested catalog product to link.
    script: list.py
    argv: [add]
    parameters:
      type: object
      properties:
        text: {type: string, description: "Item as the user said it, e.g. 'leche entera 1L'."}
        quantity: {type: integer, description: Units, default 1.}
        canonical_id: {type: integer, description: Catalog product id to link, if known.}
      required: [text]
  - name: shopping_list_remove
    description: Remove items from the shopping list by id (ids from shopping_list_show).
    script: list.py
    argv: [remove]
    parameters:
      type: object
      properties:
        ids: {type: string, description: "Comma-separated item ids, e.g. '3,7'."}
      required: [ids]
  - name: shopping_list_clear
    description: Empty the whole shopping list. The tool asks the user to confirm, so call it directly.
    script: list.py
    argv: [clear]
    confirm: true
    parameters: {type: object, properties: {}}
  - name: shopping_receipt_add
    description: >-
      Register a shopping receipt (store, date, one item per purchased unit).
      Returns list items it matched, to offer removing them.
    script: receipt.py
    argv: [add]
    parameters:
      type: object
      properties:
        store:
          type: string
          description: Store name from the receipt.
        date:
          type: string
          description: Purchase date in YYYY-MM-DD format.
        items:
          type: array
          description: One entry per purchased unit. Do not include items with unknown prices.
          items:
            type: object
            properties:
              name: {type: string}
              price: {type: number}
            required: [name, price]
        total:
          type: number
          description: Receipt total if visible.
        image_path:
          type: string
          description: Absolute path under ~/.eyetor/images for the saved receipt image.
      required: [store, date, items]
  - name: shopping_receipt_reconfirm
    description: Re-read a receipt photo with the vision model when prices, total or store are missing.
    script: receipt.py
    argv: [reconfirm]
    parameters:
      type: object
      properties:
        image_path: {type: string, description: "Path from the 'Imagen guardada en:' line."}
      required: [image_path]
  - name: shopping_price_history
    description: Prices the user paid for a product, from their own registered receipts.
    script: price.py
    argv: [history]
    parameters:
      type: object
      properties:
        product: {type: string, description: Product name.}
        store: {type: string, description: Only this store.}
        days: {type: integer, description: Only the last N days.}
      required: [product]
  - name: shopping_price_cheapest
    description: Where a product was cheapest, from the user's own registered receipts (not the web).
    script: price.py
    argv: [cheapest]
    parameters:
      type: object
      properties:
        product: {type: string, description: Product name.}
        strategy:
          type: string
          enum: [last-known, min-ever, min-last-30d]
          description: "Price to compare per store: last paid (default), lowest ever, or lowest in 30 days."
      required: [product]
  - name: shopping_plan
    description: Plan how to buy the current shopping list across stores.
    script: optimize.py
    argv: [plan]
    parameters:
      type: object
      properties:
        mode:
          type: string
          enum: [cheapest, fewer-stores]
          description: Lowest total price, or fewest stores visited.
        max_stores: {type: integer, description: "fewer-stores only: maximum number of stores."}
      required: [mode]
  - name: shopping_catalog_list
    description: List catalog (canonical) products, optionally filtered.
    script: product.py
    argv: [canonical, list]
    parameters:
      type: object
      properties:
        search: {type: string, description: Text to filter by.}
  - name: shopping_catalog_create
    description: Create a catalog (canonical) product.
    script: product.py
    argv: [canonical, create]
    parameters:
      type: object
      properties:
        name: {type: string, description: "e.g. 'Leche entera 1L'."}
        category: {type: string, description: "e.g. 'lacteos'."}
      required: [name]
  - name: shopping_catalog_alias
    description: Link a receipt/list name (alias) to a catalog product.
    script: product.py
    argv: [alias, add]
    parameters:
      type: object
      properties:
        alias: {type: string, description: "Name as it appears, e.g. 'le ent 1l'."}
        canonical_id: {type: integer, description: Catalog product id.}
      required: [alias, canonical_id]
---

# shopping — compras, lista y optimización

Skill para registrar tickets de supermercado a partir de fotos, mantener una lista de la
compra y responder preguntas como "¿dónde es más barato X?" o "¿qué ruta de tiendas me
ahorra visitas?".

Toda la persistencia vive en una sola base de datos SQLite global:
`~/.eyetor/shopping.db`. Cada **unidad** comprada se almacena como una fila independiente
(si en un ticket aparecen 3 leches, son 3 filas).

## Herramientas

| Tool | Para qué |
|------|----------|
| `shopping_list_show` / `_add` / `_remove` / `_clear` | Lista de la compra (`_clear` pide confirmación) |
| `shopping_receipt_add` | Registrar un ticket |
| `shopping_receipt_reconfirm` | Releer la foto de un ticket con visión |
| `shopping_price_history` / `shopping_price_cheapest` | Historial y tienda más barata |
| `shopping_plan` | Ruta de compra: más barata o con menos tiendas |
| `shopping_catalog_list` / `_create` / `_alias` | Catálogo de productos canónicos |

## Flujo de un ticket por foto

1. El usuario manda foto a Telegram. El canal guarda la imagen en
   `~/.eyetor/images/{chat_id}_{ts}.jpg` y añade al contexto una línea
   `Imagen guardada en: <path>` junto con la descripción del modelo de visión y,
   cuando existe, el caption del usuario.

2. Extrae del texto descriptivo: `store`, `date` (formato `YYYY-MM-DD`), `items`
   (lista de `{name, price}` con **precio unitario**), `total` si aparece.
   Si un producto aparece N veces en el ticket, **emite N entradas idénticas** en
   `items` (una por unidad). Si no hay fecha visible en la imagen, busca una fecha
   completa en el caption/mensaje del usuario y úsala. Si tampoco hay fecha en el
   caption, pregunta al usuario antes de registrar.

3. Llama a `shopping_receipt_add`:

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

4. Posibles respuestas:

   - `{"ok":true, "receipt_id":N, "inserted":3, "reconcile":[...]}` → todo bien.
   - `{"ok":true, "needs_reconfirm":true, "reason":"..."}` → faltan precios o el
     total no cuadra. **No se ha insertado nada**.

5. Si necesita reconfirmación, llama a `shopping_receipt_reconfirm` con la misma
   `image_path`. Devuelve `{"ok":true, "store":..., "date":..., "total":..., "items":[...]}`.
   Vuelve al paso 3 con los datos enriquecidos.

6. Si el `reconcile` no está vacío, redacta un mensaje en lenguaje natural mostrando
   los ítems del ticket que estaban en la lista, y **pide confirmación al usuario**
   antes de borrarlos:

   > "He registrado el ticket de Mercadona (3 productos, 2,95 €).
   > He visto que tenías **Pan** y **Leche entera 1L** en la lista de la compra.
   > ¿Los marco como comprados?"

7. Solo si el usuario confirma, llama a `shopping_list_remove` con esos ids.

## Lista de la compra

Al añadir, si la herramienta devuelve `suggest_canonical` (score entre 0.70 y 0.85),
pregunta al usuario antes de enlazar con ese canonical.

## Precios y optimización

`shopping_plan` devuelve un plan agrupado por tienda con el coste total y los productos
sin historial en `missing`.

## Reglas estrictas

- **Nunca inventes precios.** Si un ítem no tiene precio extraído del texto/visión,
  no lo incluyas; deja que la herramienta pida reconfirmación.
- **Fechas siempre `YYYY-MM-DD`.** Convierte tú "12/05/2026" → "2026-05-12".
  Prioridad: imagen → caption/mensaje → preguntar al usuario. No inventes una fecha.
- **Items duplicados en un ticket: una entrada por unidad** en `items`. Se guarda
  una fila por unidad para que `MIN/AVG(price)` y las comparaciones funcionen.
- **Reconciliación con la lista: confirmación obligatoria.** No llames a
  `shopping_list_remove` sin un "sí" explícito del usuario.
- **Reconfirm sólo si la herramienta lo pide o si el usuario lo solicita.** No
  re-proceses la imagen si `shopping_receipt_add` aceptó el ticket.

## Uso manual de los scripts

Los scripts siguen funcionando a mano (`scripts/list.py show`, `scripts/price.py
cheapest --product X`, `scripts/receipt.py undo --receipt-id N`,
`scripts/product.py alias list|delete ...`; ver `--help` de cada uno). `receipt.py undo`
y `product.py canonical merge` / `alias list|delete` no tienen tool: son mantenimiento.
