import weaviate

client = weaviate.connect_to_local(port=5000, grpc_port=50051)

collection: weaviate.collections.Collection = client.collections.get("Malware1")

response = collection.query.fetch_objects(
    limit=10,
    include_vector=True
)

for o in response.objects:
    print(o.vector['opcode_vector'])

client.close()
