from os import getenv
from traceback import print_exc
from typing import Tuple, List, Dict, Any, Optional

from weaviate import connect_to_custom, connect_to_local
import numpy as np
from fastapi import HTTPException


class WeaviateClient:
    """
    Wrapper for Weaviate Client.
    Helps forward Weaviate errors to fastapi.
    """

    def __init__(self, port: int = 5000, grpc_port: int = 50051):
        self.port = port
        self.grpc_port = grpc_port
        self.client = None
        self._setup_client()
        # self._create_collection() --> removing auto-creation; no need anymore

    def _setup_client(self):
        """Initialize Weaviate client"""
        try:
            weaviate_host = getenv("WEAVIATE_HOST", "weaviate.malwareuniverse.org")
            weaviate_http_port = getenv("WEAVIATE_HTTP_PORT", 443)
            weaviate_http_secure = getenv("WEAVIATE_HTTP_SECURE", True)
            weaviate_grpc_port = getenv("WEAVIATE_GRPC_PORT", 50051)
            weaviate_grpc_secure = getenv("WEAVIATE_GRPC_SECURE", False)

            if weaviate_host:
                self.client = connect_to_custom(
                    http_host=weaviate_host,
                    http_port=int(weaviate_http_port),
                    http_secure=bool(weaviate_http_secure),
                    grpc_host=weaviate_host,
                    grpc_port=int(weaviate_grpc_port),
                    grpc_secure=bool(weaviate_grpc_secure)
                )
            else:
                self.client = connect_to_local(
                    port=self.port,
                    grpc_port=self.grpc_port,
                )
        except Exception as e:
            print(e)
            raise ConnectionError(f"Failed to connect to Weaviate: {str(e)}")

    def query_vectors(self,
                      collection_name: str,
                      query: str,
                      limit: Optional[int] = None,
                      vector_field: Optional[str] = None
                      ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if not self.client:
            raise HTTPException(status_code=500,
                                detail="Weaviate client not initialized"
                                )

        try:
            collection = self.client.collections.get(collection_name)
            vectors = []
            metadata = []

            if limit is None:
                for obj in collection.iterator(include_vector=True):
                    self._process_object(obj, vectors, metadata, vector_field)
            else:
                response = collection.query.fetch_objects(limit=limit,
                                                          include_vector=True
                                                          )
                for obj in response.objects:
                    self._process_object(obj,
                                         vectors,
                                         metadata,
                                         vector_field)

            if not vectors:
                raise HTTPException(status_code=404,
                                    detail=f"""
                                            No vectors found in collection:
                                            {collection_name}
                                            """)

            return np.array(vectors), metadata

        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500,
                                detail=f"Weaviate query failed: {str(e)}"
                                )

    def _process_object(self, obj, vectors, metadata, vector_field):
        """Extract vector and metadata from a Weaviate object."""
        if hasattr(obj, 'vector') and obj.vector:
            vector_data = self._extract_vector(obj.vector, vector_field)
            if vector_data:
                vectors.append(vector_data)
                metadata.append({
                    'uuid': str(obj.uuid),
                    'properties': obj.properties,
                    'vector_length': len(vector_data)
                })

    @staticmethod
    def _extract_vector(vector_dict: Dict,
                        vector_field: Optional[str] = None
                        ) -> Optional[List[float]]:
        """Extract vector data from Weaviate response"""
        if vector_field and vector_field in vector_dict:
            return vector_dict[vector_field]

        common_fields = ['title_vector', 'default', 'vector', 'embedding']
        for field in common_fields:
            if field in vector_dict:
                return vector_dict[field]

        if vector_dict:
            return list(vector_dict.values())[0]

        return None

    def list_collections(self) -> dict:
        """List all available collections"""
        if not self.client:
            raise HTTPException(status_code=500,
                                detail="Weaviate client not initialized"
                                )

        try:
            collections = self.client.collections.list_all()
            return collections
        except Exception as e:
            print_exc()
            raise HTTPException(status_code=500,
                                detail=f"Failed to list collections: {str(e)}"
                                )

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        if not self.client:
            raise HTTPException(status_code=500,
                                detail="Weaviate client not initialized"
                                )

        try:
            collection = self.client.collections.get(collection_name)
            sample = collection.query.fetch_objects(limit=1,
                                                    include_vector=True
                                                    )

            info = {
                "name": collection_name,
                "exists": True,
                "sample_count": 1 if sample.objects else 0
            }

            if sample.objects:
                obj = sample.objects[0]
                info["properties"] = list(obj.properties.keys()) if obj.properties else []
                info["vector_fields"] = list(obj.vector.keys()) if hasattr(obj, 'vector') and obj.vector else []
                if info["vector_fields"]:
                    first_vector = list(obj.vector.values())[0]
                    info["vector_dimension"] = len(first_vector) if first_vector else 0

            return info

        except Exception as e:
            return {
                "name": collection_name,
                "exists": False,
                "error": str(e)
            }

    def close(self):
        """Close the Weaviate client connection"""
        if self.client:
            self.client.close()
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
