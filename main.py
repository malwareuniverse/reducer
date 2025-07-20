from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import time

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import numpy as np

# Own Modules
from dimensionality_reducer import DRMethod, DRFactory
from weaviate_client import WeaviateClient

from clusterer import ClusterMethod, ClusterFactory

N_samples = 1024
D_dimensions = 1024

weaviate_client = None

# Pydantic Response Models
class WeaviateResult(BaseModel):
    """Represents a single data point with its transformed embedding and original metadata."""
    embedding: List[float]
    metadata: Dict[str, Any]

class BaseDataResponse(BaseModel):
    """Base response containing fields common to all data-returning endpoints."""
    shape: List[int]
    method_applied: str
    dr_applied: bool
    message: str

class GenerateDataResponse(BaseDataResponse):
    """Response for randomly generated data."""
    data: List[List[float]]
    data_source: str = "random"

class QueryWeaviateResponse(BaseDataResponse):
    data_source: str = "weaviate"
    collection_name: str
    original_vector_count: int
    query: str
    clustering_applied: bool
    clustering_method_applied: Optional[str] = None
    results: List[WeaviateResult]

class AvailableClusteringMethodsResponse(BaseModel):
    available_methods: List[str]
    all_methods: List[str]
class WeaviateCollectionsResponse(BaseModel):
    collections: List[str]

class AvailableMethodsResponse(BaseModel):
    available_methods: List[str]
    all_methods: List[str]

class RootResponse(BaseModel):
    status: str

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    global weaviate_client
    weaviate_client = WeaviateClient(port=5000, grpc_port=50051)
    yield
    print(weaviate_client)
    if weaviate_client:
        weaviate_client.close()

# Dependency function
def get_weaviate_client() -> Optional[WeaviateClient]:
    if weaviate_client is None:
        raise HTTPException(status_code=500, detail="Weaviate client not initialized")
    return weaviate_client

# FastAPI app with lifespan
app = FastAPI(
    title="Multi-Algorithm Dimensionality Reduction API with Weaviate",
    description=f"Generates random data or queries Weaviate, then applies various DR algorithms.",
    version="2.1.0",
    lifespan=lifespan
)


@app.get(
    "/generate_data",
    response_model=GenerateDataResponse,
    summary="Generate random data with optional dimensionality reduction",
    response_description="A JSON object containing the generated data or DR embedding."
)
async def generate_and_transform_data(
        apply_dr: bool = True,
        dr_method: DRMethod = DRMethod.PACMAP,
        n_components: int = 3,
        pacmap_n_neighbors: int = 10,
        pacmap_mn_ratio: float = 0.5,
        pacmap_fp_ratio: float = 2.0,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = "euclidean",
        trimap_n_inliers: int = 10,
        trimap_n_outliers: int = 5,
        trimap_n_random: int = 5,
        verbose: bool = True
) -> GenerateDataResponse:
    """Generate random dataset and optionally apply dimensionality reduction."""

    X_test_data = np.random.randn(N_samples, D_dimensions)

    # If apply_dr is False, return original data without DR
    if not apply_dr:
        data_to_return = X_test_data
        dr_applied_result = False
        dr_error = None
        method_name = "None"
    else:
        # Apply dimensionality reduction
        data_to_return, dr_applied_result, dr_error, method_name = _apply_dimensionality_reduction(
            X_test_data, apply_dr, dr_method, n_components,
            pacmap_n_neighbors, pacmap_mn_ratio, pacmap_fp_ratio,
            umap_n_neighbors, umap_min_dist, umap_metric,
            trimap_n_inliers, trimap_n_outliers, trimap_n_random,
            verbose
        )

    result_list = data_to_return.tolist()
    message = _get_response_message(dr_applied_result, dr_error, method_name, dr_method, n_components)

    return GenerateDataResponse(
        shape=list(data_to_return.shape),
        method_applied=method_name,
        dr_applied=dr_applied_result,
        data=result_list,
        data_source="random",
        message=message
    )


# In main.py, replace the entire `/query_weaviate` endpoint function with this updated version:

@app.get(
    "/query_weaviate",
    response_model=QueryWeaviateResponse,
    summary="Query Weaviate, apply clustering, and apply dimensionality reduction",
    response_description="A JSON object containing Weaviate data with optional clustering and DR."
)
async def query_and_transform_weaviate_data(
        # Weaviate Query Parameters
        collection_name: str,
        query: str = "",
        limit: int = 100,
        vector_field: Optional[str] = None,
        # Clustering Parameters
        apply_clustering: bool = True,
        cluster_method: ClusterMethod = ClusterMethod.HDBSCAN,
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: Optional[int] = None,
        hdbscan_metric: str = "euclidean",
        # Dimensionality Reduction Parameters
        apply_dr: bool = True,
        dr_method: DRMethod = DRMethod.PACMAP,
        n_components: int = 3,
        # PaCMAP Parameters
        pacmap_n_neighbors: int = 10,
        pacmap_mn_ratio: float = 0.5,
        pacmap_fp_ratio: float = 2.0,
        # UMAP Parameters
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = "euclidean",
        # TriMAP Parameters
        trimap_n_inliers: int = 10,
        trimap_n_outliers: int = 5,
        trimap_n_random: int = 5,
        # General
        verbose: bool = False,
        client: WeaviateClient = Depends(get_weaviate_client)
) -> QueryWeaviateResponse:
    """
    Query Weaviate for vectors, then optionally apply clustering and/or dimensionality reduction.

    - **Clustering** is performed on the original high-dimensional vectors.
    - **Dimensionality Reduction** is also performed on the original vectors.
    """

    X_weaviate_data, metadata = client.query_vectors(
        collection_name=collection_name,
        query=query,
        limit=limit,
        vector_field=vector_field
    )

    cluster_labels, clustering_applied, clustering_error, clustering_method_name = _apply_clustering(
        X_weaviate_data, apply_clustering, cluster_method,
        hdbscan_min_cluster_size, hdbscan_min_samples, hdbscan_metric, verbose
    )

    reduced_embeddings, dr_applied, dr_error, dr_method_name = _apply_dimensionality_reduction(
        X_weaviate_data, apply_dr, dr_method, n_components,
        pacmap_n_neighbors, pacmap_mn_ratio, pacmap_fp_ratio,
        umap_n_neighbors, umap_min_dist, umap_metric,
        trimap_n_inliers, trimap_n_outliers, trimap_n_random,
        verbose
    )

    # --- STEP 3: COMBINE RESULTS ---
    if clustering_applied and cluster_labels is not None:
        for i, meta in enumerate(metadata):
            meta['cluster_label'] = int(cluster_labels[i])

    result_points = reduced_embeddings.tolist()
    combined_results = [
        WeaviateResult(embedding=point, metadata=meta)
        for point, meta in zip(result_points, metadata)
    ]

    dr_message = _get_response_message(dr_applied, dr_error, dr_method_name, dr_method, n_components)

    cluster_message = ""
    if not apply_clustering:
        cluster_message = "Clustering not applied. "
    elif clustering_error:
        cluster_message = f"{cluster_method.value} clustering failed: {clustering_error}. "
    elif clustering_applied and cluster_labels is not None:
        num_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        num_noise = np.sum(cluster_labels == -1)
        cluster_message = f"{clustering_method_name} found {num_clusters} clusters and {num_noise} noise points. "

    final_message = cluster_message + dr_message

    return QueryWeaviateResponse(
        shape=list(reduced_embeddings.shape),
        method_applied=dr_method_name,
        dr_applied=dr_applied,
        clustering_applied=clustering_applied,
        clustering_method_applied=clustering_method_name if clustering_applied else None,
        results=combined_results,
        data_source="weaviate",
        collection_name=collection_name,
        query=query,
        original_vector_count=len(metadata),
        message=final_message
    )

@app.get(
    "/weaviate_collections",
    response_model=WeaviateCollectionsResponse
)
async def list_weaviate_collections(
        client: WeaviateClient = Depends(get_weaviate_client)
) -> WeaviateCollectionsResponse:
    """List all available Weaviate collections"""
    try:
        collections = client.list_collections()
        print(collections)
        return WeaviateCollectionsResponse(collections=collections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@app.get("/weaviate_collection_info/{collection_name}")
async def get_weaviate_collection_info(
        collection_name: str,
        client: WeaviateClient = Depends(get_weaviate_client)
):
    """Get information about a specific Weaviate collection"""
    try:
        info = client.get_collection_info(collection_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

@app.get(
    "/available_clustering_methods",
    response_model=AvailableClusteringMethodsResponse,
    summary="List available clustering algorithms"
)
async def get_available_clustering_methods() -> AvailableClusteringMethodsResponse:
    return AvailableClusteringMethodsResponse(
        available_methods=ClusterFactory.get_available_methods(),
        all_methods=[method.value for method in ClusterMethod]
    )

@app.get("/", response_model=RootResponse)
async def read_root() -> RootResponse:
    return RootResponse(status="Multi-Algorithm DR & Clustering API with Weaviate is running")

@app.get(
    "/available_methods",
    response_model=AvailableMethodsResponse
)
async def get_available_methods() -> AvailableMethodsResponse:
    """Get list of available dimensionality reduction methods"""
    return AvailableMethodsResponse(
        available_methods=DRFactory.get_available_methods(),
        all_methods=[method.value for method in DRMethod]
    )

@app.get(
    "/",
    response_model=RootResponse
)
async def read_root() -> RootResponse:
    return RootResponse(status="Multi-Algorithm Dimensionality Reduction API with Weaviate is running")

# Helper functions
def _apply_dimensionality_reduction(X_data, apply_dr, dr_method, n_components, *args):
    """Apply dimensionality reduction to data"""
    if not apply_dr:
        return X_data, False, None, "None"

    if len(X_data) <= 12:  # at least n = 12
        raise HTTPException(status_code=400, detail="Database must at least include 12 entries")

    if n_components < 1:
        raise HTTPException(status_code=400, detail="n_components must be at least 1")
    if n_components > X_data.shape[1]:
        n_components = min(n_components, X_data.shape[1])
        print(f"Warning: n_components reduced to {n_components} as it exceeded input dimensions.")

    available_methods = DRFactory.get_available_methods()
    if dr_method.value not in available_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Method {dr_method.value} is not available. Available methods: {available_methods}"
        )

    try:
        (pacmap_n_neighbors, pacmap_mn_ratio, pacmap_fp_ratio,
         umap_n_neighbors, umap_min_dist, umap_metric,
         trimap_n_inliers, trimap_n_outliers, trimap_n_random, verbose) = args

        kwargs = {"verbose": verbose}

        if dr_method == DRMethod.PACMAP:
            kwargs.update({
                "n_neighbors": pacmap_n_neighbors,
                "MN_ratio": pacmap_mn_ratio,
                "FP_ratio": pacmap_fp_ratio,
                "apply_pca": True,
                "init": "pca"
            })
        elif dr_method == DRMethod.UMAP:
            kwargs.update({
                "n_neighbors": umap_n_neighbors,
                "min_dist": umap_min_dist,
                "metric": umap_metric
            })
        elif dr_method == DRMethod.TRIMAP:
            kwargs.update({
                "n_inliers": trimap_n_inliers,
                "n_outliers": trimap_n_outliers,
                "n_random": trimap_n_random
            })

        reducer = DRFactory.create_reducer(dr_method, n_components=n_components, **kwargs)
        method_name = reducer.name

        start_time = time.time()
        X_transformed = reducer.fit_transform(X_data)
        end_time = time.time()
        print(f"{method_name} finished in {end_time - start_time:.2f} seconds.")

        return X_transformed, True, None, method_name

    except Exception as e:
        print(f"\nAn error occurred during {dr_method.value} execution: {e}")
        return X_data, False, str(e), "None"

def _get_response_message(dr_applied, dr_error, method_name, dr_method, n_components):
    """Get appropriate response message"""
    if dr_error:
        return f"{dr_method.value} failed: {dr_error}. Returning original data."
    elif dr_applied:
        return f"{method_name} applied successfully, returning {n_components}-component embedding."
    else:
        return "Original data returned."


def _apply_clustering(X_data, apply_clustering, cluster_method, *args):
    """Apply clustering to high-dimensional data."""
    if not apply_clustering:
        return None, False, None, "None"

    # Unpack args
    (hdbscan_min_cluster_size, hdbscan_min_samples, hdbscan_metric, verbose) = args

    if len(X_data) < hdbscan_min_cluster_size:
        error_msg = f"Not enough data points ({len(X_data)}) to cluster. HDBSCAN requires at least 'min_cluster_size' points ({hdbscan_min_cluster_size})."
        return None, False, error_msg, "None"

    available_methods = ClusterFactory.get_available_methods()
    if cluster_method.value not in available_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Clustering method {cluster_method.value} is not available. Available: {available_methods}"
        )

    try:
        kwargs = {}
        if cluster_method == ClusterMethod.HDBSCAN:
            kwargs.update({
                "min_cluster_size": hdbscan_min_cluster_size,
                "min_samples": hdbscan_min_samples,
                "metric": hdbscan_metric,
            })

        clusterer = ClusterFactory.create_clusterer(cluster_method, **kwargs)
        method_name = clusterer.name

        start_time = time.time()
        print(f"Starting clustering with {method_name}...")
        labels = clusterer.fit_predict(X_data)
        end_time = time.time()
        print(f"Clustering with {method_name} finished in {end_time - start_time:.2f} seconds.")

        return labels, True, None, method_name

    except Exception as e:
        error_str = f"An error occurred during {cluster_method.value} clustering: {e}"
        print(f"\n{error_str}")
        return None, False, str(e), "None"
