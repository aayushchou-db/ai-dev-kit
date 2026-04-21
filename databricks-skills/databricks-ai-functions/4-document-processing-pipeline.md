# Document Processing Pipeline with AI Functions

End-to-end patterns for building batch document processing pipelines using AI Functions in a Lakeflow Declarative Pipeline (DLT). Covers function selection, `config.yml` centralization, error handling, and guidance on near-real-time variants with DSPy or LangChain.

> For workflow migration context (e.g., migrating from n8n, LangChain, or other orchestration tools), see the companion skill `n8n-to-databricks`.

---

## Function Selection for Document Pipelines

When processing documents with AI Functions, apply this order of preference for each stage:

| Stage | Preferred function | Fall back to `ai_query` when... |
|---|---|---|
| Parse binary docs (PDF, DOCX, PPTX, images, TIFF) | `ai_parse_document` | Need image-level reasoning beyond built-in descriptions |
| Extract flat or nested fields from text | `ai_extract` v2 (schema-based) | Schema exceeds 128 fields or 7 nesting levels |
| Extract arrays of objects (e.g., line items) | `ai_extract` v2 with `"type": "array"` | Array items exceed 7 nesting levels |
| Classify document type or status | `ai_classify` | More than 20 categories |
| Score item similarity / matching | `ai_similarity` | Need cross-document reasoning |
| Summarize long sections | `ai_summarize` | — |
| Complex multi-step reasoning | `ai_query` with `responseFormat` | This is the intended use case |

> **v2 changed the boundary.** In v1, any nested array required `ai_query`. In v2, `ai_extract` handles objects, arrays of objects, enums, and typed fields up to 7 nesting levels and 128 fields. Reserve `ai_query` for schemas that exceed these limits or require multi-step reasoning.

---

## `ai_parse_document` Quick Reference

> **Official docs:** https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document

**Requires:** DBR 17.1+ | Serverless: environment version 3+

```sql
ai_parse_document(content [, options MAP<STRING, STRING>]) → VARIANT
```

**Supported formats:** PDF, JPG/JPEG, PNG, TIFF/TIF, DOC/DOCX, PPT/PPTX

**Limits:** 500 pages max per document, 100 MB file size max

**Options:**

| Key | Values | Description |
|-----|--------|-------------|
| `version` | `'2.0'` | Output schema version |
| `imageOutputPath` | Volume path | Save rendered page images to UC volume |
| `descriptionElementTypes` | `''`, `'figure'`, `'*'` (default) | Control AI-generated descriptions |
| `pageRange` | e.g. `'1,3,5-10'` | 1-indexed page subset (must stay within 500-page limit) |

**Output schema:**

```
document
├── pages[]              -- {id: INT, image_uri: STRING}
└── elements[]           -- extracted content
    ├── id               -- INT, 0-based position
    ├── type             -- "text", "table", "figure", "title", "caption",
    │                       "section_header", "page_header", "page_footer",
    │                       "page_number", "footnote"
    ├── content          -- STRING (HTML for tables)
    ├── confidence       -- DOUBLE, extraction reliability score
    ├── bbox[]           -- {coord: [INT], page_id: INT}
    └── description      -- STRING, AI-generated
metadata
├── id, version
└── file_metadata        -- {file_path, file_name, file_size, file_modification_time}
error_status[]           -- {error_message: STRING, page_id: INT}
```

**VARIANT access paths:**

```sql
-- Elements are at document.elements, NOT nested under pages
parsed:document.elements          -- array of all elements
parsed:document.pages             -- array of page metadata
parsed:error_status               -- array of per-page errors (NULL if no errors)
parsed:metadata.file_metadata     -- file info
```

---

## Centralized Configuration (`config.yml`)

**Always centralize model names, volume paths, and prompts in a `config.yml`.** This makes model swaps a one-line change and keeps pipeline code free of hardcoded strings.

```yaml
# config.yml
models:
  default: "databricks-claude-sonnet-4"
  mini:    "databricks-meta-llama-3-1-8b-instruct"
  vision:  "databricks-llama-4-maverick"

catalog:
  name:   "my_catalog"
  schema: "document_processing"

volumes:
  input: "/Volumes/my_catalog/document_processing/landing/"
  tmp:   "/Volumes/my_catalog/document_processing/tmp/"

output_tables:
  results: "my_catalog.document_processing.processed_docs"
  errors:  "my_catalog.document_processing.processing_errors"

prompts:
  extract_complex: |
    Extract the requested fields and return ONLY valid JSON.
    Return null for missing fields.

  classify_doc: |
    Classify this document into exactly one category.
```

```python
# config_loader.py
import yaml

def load_config(path: str = "config.yml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

CFG           = load_config()
ENDPOINT      = CFG["models"]["default"]
ENDPOINT_MINI = CFG["models"]["mini"]
VOLUME_INPUT  = CFG["volumes"]["input"]
```

---

## Batch Pipeline — Lakeflow Declarative Pipeline

Each logical step in your document workflow maps to a `@dlt.table` stage. Data flows through Delta tables between stages.

```
[Landing Volume]  →  Stage 1: ai_parse_document
                  →  Stage 2: ai_classify (document type)
                  →  Stage 3: ai_extract v2 (flat + nested fields in one call)
                  →  Stage 4: ai_similarity (item matching)
                  →  Stage 5: Final Delta output table
```

### `pipeline.py`

```python
import dlt
import yaml
from pyspark.sql.functions import expr, col, from_json

CFG      = yaml.safe_load(open("/Workspace/path/to/config.yml"))
ENDPOINT = CFG["models"]["default"]
VOL_IN   = CFG["volumes"]["input"]


# ── Stage 1: Parse binary documents ──────────────────────────────────────────
# Preferred: ai_parse_document — no model selection, no ai_query needed
# Note: elements are at document.elements, not under pages

@dlt.table(comment="Parsed document text from all file types in the landing volume")
def raw_parsed():
    return (
        spark.read.format("binaryFile").load(VOL_IN)
        .withColumn("doc", expr("ai_parse_document(content, MAP('version', '2.0'))"))
        .selectExpr(
            "path",
            "doc",
            "doc:error_status AS parse_errors",
        )
        .filter("parse_errors IS NULL")
    )


# ── Stage 2: Classify document type ──────────────────────────────────────────
# Preferred: ai_classify — cheap, no endpoint selection
# Extract text from elements for classification

@dlt.table(comment="Document type classification")
def classified_docs():
    return (
        dlt.read("raw_parsed")
        .withColumn("text_content", expr("""
            concat_ws('\\n', transform(
                try_cast(doc:document:elements AS ARRAY),
                e -> try_cast(e:content AS STRING)
            ))
        """))
        .withColumn(
            "doc_type",
            expr("ai_classify(text_content, array('invoice', 'purchase_order', 'receipt', 'contract', 'other'))")
        )
    )


# ── Stage 3: Field extraction with ai_extract v2 ─────────────────────────────
# v2 handles BOTH flat fields AND arrays of objects in a single call.
# Pass the VARIANT doc directly — v2 accepts VARIANT input from ai_parse_document.
# MUST include MAP('version', '2.0') to activate v2.
# Return is VARIANT: {response: {...}, error_message: null}

@dlt.table(comment="Extracted invoice fields — header + line items via ai_extract v2")
def extracted_invoices():
    return (
        dlt.read("classified_docs")
        .filter("doc_type = 'invoice'")
        .withColumn(
            "result",
            expr("""
                ai_extract(
                    doc,
                    '{
                      "invoice_number": {"type": "string"},
                      "vendor_name": {"type": "string"},
                      "issue_date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                      "total_amount": {"type": "number"},
                      "tax_id": {"type": "string", "description": "Vendor tax identifier"},
                      "currency": {
                        "type": "enum",
                        "labels": ["USD", "EUR", "GBP", "CAD", "AUD"],
                        "description": "Invoice currency code"
                      },
                      "line_items": {
                        "type": "array",
                        "description": "Itemized line items on the invoice",
                        "items": {
                          "type": "object",
                          "properties": {
                            "item_code": {"type": "string"},
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                            "total": {"type": "number"}
                          }
                        }
                      }
                    }',
                    MAP('version', '2.0', 'instructions', 'These are vendor invoices. Extract all header fields and line items.')
                )
            """)
        )
        .selectExpr(
            "path",
            "doc_type",
            "text_content",
            "result:response AS extracted",
            "result:error_message::STRING AS extract_error"
        )
    )


# ── Stage 3 fallback: ai_query for schemas exceeding v2 limits ───────────────
# Use ONLY when your schema exceeds 128 fields, 7 nesting levels, or 500 enum
# values. For most invoice/contract schemas, ai_extract v2 above is sufficient.
#
# @dlt.table(comment="Complex extraction via ai_query — v2 limits exceeded")
# def extracted_complex():
#     return (
#         dlt.read("classified_docs")
#         .filter("doc_type = 'contract'")
#         .withColumn(
#             "ai_response",
#             expr(f"""
#                 ai_query(
#                     '{ENDPOINT}',
#                     concat('Extract contract fields as JSON:\\n\\n', LEFT(text_content, 6000)),
#                     responseFormat => '{{"type":"json_object"}}',
#                     failOnError     => false
#                 )
#             """)
#         )
#         .withColumn("contract", from_json(col("ai_response.response"), "STRUCT<...>"))
#         .select("path", "doc_type", "contract", col("ai_response.error").alias("extraction_error"))
#     )


# ── Stage 4: Similarity matching ─────────────────────────────────────────────
# Preferred: ai_similarity for fuzzy matching between extracted fields
# Note: access v2 results via VARIANT path (extracted:field), not STRUCT dot-notation

@dlt.table(comment="Vendor name similarity vs reference master data")
def vendor_matched():
    extracted = dlt.read("extracted_invoices").filter("extract_error IS NULL")
    vendors = spark.table("my_catalog.document_processing.vendor_master").select("vendor_id", "vendor_name")

    return (
        extracted.crossJoin(vendors)
        .withColumn(
            "name_similarity",
            expr("ai_similarity(extracted:vendor_name::STRING, vendor_name)")
        )
        .filter("name_similarity > 0.80")
        .orderBy("name_similarity", ascending=False)
    )


# ── Stage 5: Final output + error sidecar ────────────────────────────────────
# Access all fields through the VARIANT extracted column using path syntax

@dlt.table(
    comment="Final processed documents ready for downstream consumption",
    table_properties={"delta.enableChangeDataFeed": "true"},
)
def processed_docs():
    return (
        dlt.read("extracted_invoices")
        .filter("extract_error IS NULL")
        .selectExpr(
            "path",
            "doc_type",
            "extracted:invoice_number::STRING AS invoice_number",
            "extracted:vendor_name::STRING AS vendor_name",
            "extracted:issue_date::STRING AS issue_date",
            "extracted:total_amount::DOUBLE AS total_amount",
            "extracted:currency::STRING AS currency",
            "extracted:line_items AS items",
        )
    )


@dlt.table(comment="Rows that failed at any extraction stage — review and reprocess")
def processing_errors():
    return (
        dlt.read("extracted_invoices")
        .filter("extract_error IS NOT NULL")
        .select("path", "doc_type", col("extract_error").alias("error"))
    )
```

---

## Custom RAG Pipeline — Parse → Chunk → Index → Query

When the goal is retrieval-augmented generation rather than field extraction, use this pipeline to parse documents, chunk them into a Delta table, and index with Vector Search.

### Step 1 — Parse and Chunk into a Delta Table

`ai_parse_document` returns a VARIANT. Use `variant_get` with an explicit `ARRAY<VARIANT>` cast before calling `explode`, since `explode()` does not accept raw VARIANT values.

```sql
CREATE OR REPLACE TABLE catalog.schema.parsed_chunks AS
WITH parsed AS (
  SELECT
    path,
    ai_parse_document(content, MAP('version', '2.0')) AS doc
  FROM read_files('/Volumes/catalog/schema/volume/docs/', format => 'binaryFile')
  WHERE ai_parse_document(content, MAP('version', '2.0')):error_status IS NULL
),
elements AS (
  SELECT
    path,
    explode(variant_get(doc, '$.document.elements', 'ARRAY<VARIANT>')) AS element
  FROM parsed
)
SELECT
  md5(concat(path, variant_get(element, '$.content', 'STRING'))) AS chunk_id,
  path AS source_path,
  variant_get(element, '$.content', 'STRING') AS content,
  variant_get(element, '$.type', 'STRING') AS element_type,
  variant_get(element, '$.confidence', 'DOUBLE') AS confidence,
  current_timestamp() AS parsed_at
FROM elements
WHERE variant_get(element, '$.content', 'STRING') IS NOT NULL
  AND length(trim(variant_get(element, '$.content', 'STRING'))) > 10;
```

### Step 1a (Production) — Incremental Parsing with Structured Streaming

For production pipelines where new documents arrive over time, use Structured Streaming with checkpoints for exactly-once processing. Each run processes only new files, then stops with `trigger(availableNow=True)`.

See the official bundle example:
[databricks/bundle-examples/contrib/job_with_ai_parse_document](https://github.com/databricks/bundle-examples/tree/main/contrib/job_with_ai_parse_document)

**Stage 1 — Parse raw documents (streaming):**

```python
from pyspark.sql.functions import col, current_timestamp, expr

files_df = (
    spark.readStream.format("binaryFile")
    .option("pathGlobFilter", "*.{pdf,jpg,jpeg,png,tiff,tif,docx,pptx}")
    .option("recursiveFileLookup", "true")
    .load("/Volumes/catalog/schema/volume/docs/")
)

parsed_df = (
    files_df
    .repartition(8, expr("crc32(path) % 8"))
    .withColumn("doc", expr("""
        ai_parse_document(content, MAP(
            'version', '2.0',
            'descriptionElementTypes', '*'
        ))
    """))
    .withColumn("parsed_at", current_timestamp())
    .select("path", "doc", "parsed_at")
)

(
    parsed_df.writeStream.format("delta")
    .outputMode("append")
    .option("checkpointLocation", "/Volumes/catalog/schema/checkpoints/01_parse")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable("catalog.schema.parsed_documents_raw")
)
```

**Stage 2 — Extract text from parsed VARIANT (streaming):**

Uses `transform()` to extract element content from the VARIANT array, and `try_cast` for safe access. Error rows are preserved but flagged.

```python
from pyspark.sql.functions import col, concat_ws, expr, lit, when

parsed_stream = spark.readStream.format("delta").table("catalog.schema.parsed_documents_raw")

text_df = (
    parsed_stream
    .withColumn("text",
        when(
            expr("doc:error_status IS NOT NULL"), lit(None)
        ).otherwise(
            concat_ws("\n\n", expr("""
                transform(
                    try_cast(doc:document:elements AS ARRAY),
                    element -> try_cast(element:content AS STRING)
                )
            """))
        )
    )
    .withColumn("error_status", expr("try_cast(doc:error_status AS STRING)"))
    .select("path", "text", "error_status", "parsed_at")
)

(
    text_df.writeStream.format("delta")
    .outputMode("append")
    .option("checkpointLocation", "/Volumes/catalog/schema/checkpoints/02_text")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable("catalog.schema.parsed_documents_text")
)
```

**Stage 3 (optional) — Extract structured fields with ai_extract v2 (streaming):**

```python
extract_stream = (
    spark.readStream.format("delta")
    .table("catalog.schema.parsed_documents_raw")
    .filter("doc:error_status IS NULL")
    .withColumn("result", expr("""
        ai_extract(
            doc,
            '{"invoice_id": {"type": "string"}, "vendor_name": {"type": "string"}, "total_amount": {"type": "number"}}',
            MAP('version', '2.0')
        )
    """))
    .selectExpr(
        "path",
        "result:response AS extracted",
        "result:error_message::STRING AS extract_error",
        "parsed_at"
    )
)

(
    extract_stream.writeStream.format("delta")
    .outputMode("append")
    .option("checkpointLocation", "/Volumes/catalog/schema/checkpoints/03_extract")
    .trigger(availableNow=True)
    .toTable("catalog.schema.extracted_documents")
)
```

Key techniques:
- **`repartition` by file hash** — parallelizes `ai_parse_document` across workers
- **`trigger(availableNow=True)`** — processes all pending files then stops (batch-like)
- **Checkpoints** — exactly-once guarantee; no re-parsing on re-runs
- **`transform()` + `try_cast`** — safer than `explode` + `variant_get` for text extraction
- **Separate stages with independent checkpoints** — parse, text extraction, and field extraction can fail/retry independently
- **VARIANT composability** — Stage 3 passes the `doc` VARIANT directly to `ai_extract` v2 without intermediate text flattening

### Step 1b — Enable Change Data Feed

Required for Vector Search Delta Sync:

```sql
ALTER TABLE catalog.schema.parsed_chunks
SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
```

### Step 2 — Create a Vector Search Index and Query It

Use the **[databricks-vector-search](../databricks-vector-search/SKILL.md)** skill to create a Delta Sync index on the chunked table and query it. Ensure CDF is enabled first (Step 1b above).

### RAG-Specific Issues

| Issue | Solution |
|-------|----------|
| `explode()` fails with VARIANT | `explode()` requires ARRAY, not VARIANT. Use `variant_get(doc, '$.document.elements', 'ARRAY<VARIANT>')` to cast before exploding |
| Short/noisy chunks | Filter with `length(trim(...)) > 10` — parsing produces tiny fragments (page numbers, headers) that pollute the index |
| Low-confidence elements | Filter on `confidence` field: `variant_get(element, '$.confidence', 'DOUBLE') > 0.5` |
| Re-parsing unchanged documents | Use Structured Streaming with checkpoints — see Step 1a above |
| Large PDFs exceed 500-page limit | Use `pageRange` option: `MAP('version', '2.0', 'pageRange', '1-100')` and process in chunks |
| File exceeds 100 MB | Split the file before ingestion or use `pageRange` to process subsets |
| Region not supported | US/EU regions only, or enable cross-geography routing |

---

## Near-Real-Time Variant — DSPy + MLflow Agent

When the pipeline must respond in seconds (triggered by a user action, API call, or form submission), use DSPy with an MLflow ChatAgent instead of a DLT pipeline.

**When to use DSPy vs LangChain:**

| Scenario | Stack |
|---|---|
| Fixed pipeline steps, well-defined I/O, want prompt optimization | **DSPy** |
| Needs tool-calling, memory, or multi-agent coordination | **LangChain LCEL** + MLflow ChatAgent |
| Single LLM call, simple task | Direct AI Function or `ai_query` in a notebook |

### DSPy Signatures (replace LangChain agent system prompts)

```python
# pip install dspy-ai mlflow databricks-sdk
import dspy, yaml

CFG = yaml.safe_load(open("config.yml"))
lm = dspy.LM(
    model=f"databricks/{CFG['models']['default']}",
    api_base="https://<workspace-host>/serving-endpoints",
    api_key=dbutils.secrets.get("scope", "databricks-token"),
)
dspy.configure(lm=lm)


class ExtractInvoiceHeader(dspy.Signature):
    """Extract invoice header fields from document text."""
    document_text:  str = dspy.InputField(desc="Raw text from the document")
    invoice_number: str = dspy.OutputField(desc="Invoice number, or null")
    vendor_name:    str = dspy.OutputField(desc="Vendor/supplier name, or null")
    issue_date:     str = dspy.OutputField(desc="Date as dd/mm/yyyy, or null")
    total_amount:  float = dspy.OutputField(desc="Total amount as float, or null")


class ClassifyDocument(dspy.Signature):
    """Classify a document into one of the provided categories."""
    document_text: str = dspy.InputField()
    category:      str = dspy.OutputField(
        desc="One of: invoice, purchase_order, receipt, contract, other"
    )


class DocumentPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(ClassifyDocument)
        self.extract  = dspy.Predict(ExtractInvoiceHeader)

    def forward(self, document_text: str):
        doc_type = self.classify(document_text=document_text).category
        if doc_type == "invoice":
            header = self.extract(document_text=document_text)
            return {"doc_type": doc_type, "header": header.__dict__}
        return {"doc_type": doc_type, "header": None}


pipeline = DocumentPipeline()
```

### Wrap and Register with MLflow

```python
import mlflow, json

class DSPyDocumentAgent(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import dspy, yaml
        cfg = yaml.safe_load(open(context.artifacts["config"]))
        lm = dspy.LM(model=f"databricks/{cfg['models']['default']}")
        dspy.configure(lm=lm)
        self.pipeline = DocumentPipeline()

    def predict(self, context, model_input):
        text = model_input.iloc[0]["document_text"]
        return json.dumps(self.pipeline(document_text=text), ensure_ascii=False)

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="document_agent",
        python_model=DSPyDocumentAgent(),
        artifacts={"config": "config.yml"},
        registered_model_name="my_catalog.document_processing.document_agent",
    )
```

---

## Tips

1. **Parse first, enrich second** — always run `ai_parse_document` as the first stage. Feed its output to task-specific functions; never pass raw binary to `ai_query`.
2. **Use ai_extract v2 for flat AND nested fields** — v2 handles objects, arrays of objects, enums, and typed fields. Reserve `ai_query` for schemas exceeding 128 fields or 7 nesting levels.
3. **Always pass `MAP('version', '2.0')`** — the schema format alone does not activate v2. The version option is the control mechanism.
4. **Access v2 results through the response envelope** — `result:response.field_name::TYPE`, not `result.field_name`. Check `result:error_message IS NULL` before trusting the response.
5. **Pass VARIANT directly when composing** — `ai_extract` v2 accepts VARIANT input from `ai_parse_document`, preserving structural context that gets lost when flattening to text.
6. **`failOnError => false` is mandatory in batch `ai_query` calls** — write errors to a sidecar `_errors` table rather than crashing the pipeline.
7. **Truncate before sending to `ai_query`** — use `LEFT(text, 6000)` or chunk long documents to stay within context window limits.
8. **Use `pageRange` for large documents** — `ai_parse_document` has a 500-page, 100 MB limit. Process large PDFs in page-range chunks.
9. **Filter low-confidence elements** — `ai_parse_document` returns a `confidence` score per element; use it to filter noisy extractions.
10. **Prompts belong in `config.yml`** — never hardcode prompt strings in pipeline code. A prompt change should be a config change, not a code change.
11. **DSPy for agents** — when migrating from LangChain agent-based tools, DSPy typed `Signature` classes give you structured I/O contracts, testability, and optional prompt compilation/optimization.
