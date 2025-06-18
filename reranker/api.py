"""
FastAPI endpoints for reranker service
"""

import os
import json
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field
from .service import RerankerService


# 데이터 모델 정의
class PassageModel(BaseModel):
    """Single passage model"""
    passage_id: Optional[Any] = None
    doc_id: Optional[str] = None
    text: str
    score: Optional[float] = None
    position: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResultModel(BaseModel):
    """Search result containing multiple passages"""
    query: str
    results: List[PassageModel]
    total: Optional[int] = None
    reranked: Optional[bool] = False


class RerankerResponseModel(BaseModel):
    """Response model for reranker API"""
    query: str
    results: List[PassageModel]
    total: int
    reranked: bool = True


# API 생성
app = FastAPI(
    title="Reranker API",
    description="API for reranking search results from vector database",
    version="0.1.0"
)

# 서비스 인스턴스 생성
reranker_service = None


def get_reranker_service():
    """Get reranker service instance"""
    global reranker_service
    if reranker_service is None:
        config_path = os.environ.get("RERANKER_CONFIG")
        reranker_service = RerankerService(config_path)
    return reranker_service


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "reranker"}


@app.post("/rerank", response_model=RerankerResponseModel)
async def rerank_passages(
    data: SearchResultModel,
    top_k: Optional[int] = Query(None, description="Number of top results to return"),
    service: RerankerService = Depends(get_reranker_service)
):
    """
    Rerank search results based on query relevance
    
    Args:
        data: Search result containing query and passages to rerank
        top_k: Number of top results to return
        
    Returns:
        Reranked search results
    """
    try:
        # Convert to plain dict
        results_dict = data.dict()
        
        # Process search results - top_k 값이 있으면 정수로 변환
        if top_k is not None:
            top_k = int(top_k)
            
        reranked = service.process_search_results(data.query, results_dict, top_k)
        
        # Set total value
        reranked["total"] = len(reranked["results"])
        
        return reranked
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@app.post("/hybrid-rerank", response_model=RerankerResponseModel)
async def hybrid_rerank(
    data: SearchResultModel,
    top_k: Optional[int] = Query(None, description="Number of top results to return"),
    mrc_weight: Optional[float] = Query(0.7, description="Weight for MRC scores in hybrid reranking (0-1)"),
    service: RerankerService = Depends(get_reranker_service)
):
    """
    Rerank search results using the most appropriate method based on available models
    
    This endpoint will:
    1. Check if both FlashRank and MRC are initialized, and use hybrid reranking if both are available
    2. Use FlashRank only if MRC is not available
    3. Use MRC only if FlashRank is not available
    4. Return original results if neither is available
    
    Args:
        data: Search result containing query and passages to rerank
        top_k: Number of top results to return
        mrc_weight: Weight for MRC scores in hybrid reranking (0-1)
        
    Returns:
        Reranked search results
    """
    try:
        # Convert to plain dict
        results_dict = data.dict()
        
        # top_k 값이 있으면 정수로 변환
        if top_k is not None:
            top_k = int(top_k)
        
        # 임시로 MRC 가중치 설정 (실제 서비스에서는 이 값이 유지되어야 함)
        if hasattr(service, 'hybrid_weight_mrc'):
            original_weight = service.hybrid_weight_mrc
            service.hybrid_weight_mrc = mrc_weight
        
        # Process search results - 내부적으로 초기화 상태에 따라 적절한 방법 선택
        reranked = service.process_search_results(data.query, results_dict, top_k)
        
        # 원래 가중치 복원
        if hasattr(service, 'hybrid_weight_mrc') and 'original_weight' in locals():
            service.hybrid_weight_mrc = original_weight
        
        # Set total value
        reranked["total"] = len(reranked["results"])
        
        return reranked
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@app.post("/batch_rerank")
async def batch_rerank(
    queries: List[str] = Body(..., description="List of queries"),
    passages: List[List[Dict[str, Any]]] = Body(..., description="List of passage lists for each query"),
    top_k: Optional[int] = Query(None, description="Number of top results to return"),
    service: RerankerService = Depends(get_reranker_service)
):
    """
    Batch rerank multiple queries with their respective passages
    
    Args:
        queries: List of queries
        passages: List of passage lists corresponding to each query
        top_k: Number of top results to return for each query
        
    Returns:
        List of reranked results for each query
    """
    try:
        if len(queries) != len(passages):
            raise HTTPException(
                status_code=400, 
                detail="Number of queries must match number of passage lists"
            )
            
        results = []
        for query, passage_list in zip(queries, passages):
            reranked = service.rerank_passages(query, passage_list, top_k)
            results.append({
                "query": query,
                "results": reranked,
                "total": len(reranked),
                "reranked": True
            })
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch reranking failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 