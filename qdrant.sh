curl -X PUT "http://localhost:6333/collections/mycoll" \
 -H "Content-Type: application/json" -d '{
  "vectors": {"size": 768, "distance": "Cosine"},
  "hnsw_config": {"m": 16, "ef_construct": 256, "full_scan_threshold": 1000},
  "quantization_config": {"scalar": {"type": "int8", "always_ram": true}}
}'