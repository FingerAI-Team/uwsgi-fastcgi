"""
Client for interacting with Reranker API
"""

import json
import requests
from typing import List, Dict, Any, Optional, Union


class RerankerClient:
    """Client for reranker API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize reranker client
        
        Args:
            base_url: Base URL of reranker API
        """
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> Dict[str, str]:
        """
        Check if reranker API is healthy
        
        Returns:
            Health status
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def rerank(
        self, 
        query: str, 
        passages: List[Dict[str, Any]], 
        top_k: Optional[int] = None,
        rerank_type: Optional[str] = "auto"
    ) -> Dict[str, Any]:
        """
        Rerank passages based on query
        
        Args:
            query: Query text
            passages: List of passage dictionaries
            top_k: Number of top results to return
            rerank_type: Reranker type to use (flashrank, mrc, hybrid, auto)
            
        Returns:
            Reranked search results
        """
        payload = {
            "query": query,
            "results": passages,
            "total": len(passages),
            "reranked": False
        }
        
        params = {}
        if top_k is not None:
            params["top_k"] = top_k
        if rerank_type:
            params["type"] = rerank_type
            
        response = requests.post(
            f"{self.base_url}/rerank",
            json=payload,
            params=params
        )
        response.raise_for_status()
        results = response.json()
        
        # 기본 필드만 추출 (중요한 메타데이터 유지)
        processed_results = []
        for result in results:
            processed_result = {
                "id": result.get("id"),  # 고유 식별자 사용
                "doc_id": result["meta"].get("doc_id", ""),
                "passage_id": result["meta"].get("passage_id", ""),
                "text": result["text"],
                "score": float(result["score"])
            }

            # 메타데이터에서 original_score 보존
            if "original_score" in result["meta"]:
                metadata = {
                    "original_score": result["meta"]["original_score"],
                    "unique_id": result.get("id")  # 고유 식별자도 메타데이터에 보존
                }
                processed_result["metadata"] = metadata
            
            # MRC 관련 필드가 있으면 추가
            if "mrc_score" in result:
                processed_result["mrc_score"] = result["mrc_score"]
            if "mrc_answer" in result:
                processed_result["mrc_answer"] = result["mrc_answer"]
            if "mrc_char_ids" in result:
                processed_result["mrc_char_ids"] = result["mrc_char_ids"]
            
            # FlashRank 점수가 있으면 추가
            if "flashrank_score" in result:
                processed_result["flashrank_score"] = result["flashrank_score"]
            
            # 하이브리드 점수가 있으면 추가
            if "hybrid_score" in result:
                processed_result["hybrid_score"] = result["hybrid_score"]
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def mrc_rerank(
        self, 
        query: str, 
        passages: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Rerank passages based on query using MRC model
        
        Args:
            query: Query text
            passages: List of passage dictionaries
            top_k: Number of top results to return
            
        Returns:
            Reranked search results
        """
        payload = {
            "query": query,
            "results": passages,
            "total": len(passages),
            "reranked": False
        }
        
        params = {}
        if top_k is not None:
            params["top_k"] = top_k
            
        response = requests.post(
            f"{self.base_url}/mrc-rerank",
            json=payload,
            params=params
        )
        response.raise_for_status()
        results = response.json()
        
        # 기본 필드만 추출 (중요한 메타데이터 유지)
        processed_results = []
        for result in results:
            processed_result = {
                "id": result.get("id"),  # 고유 식별자 사용
                "doc_id": result["meta"].get("doc_id", ""),
                "passage_id": result["meta"].get("passage_id", ""),
                "text": result["text"],
                "score": float(result["score"])
            }

            # 메타데이터에서 original_score 보존
            if "original_score" in result["meta"]:
                metadata = {
                    "original_score": result["meta"]["original_score"],
                    "unique_id": result.get("id")  # 고유 식별자도 메타데이터에 보존
                }
                processed_result["metadata"] = metadata
            
            # MRC 관련 필드가 있으면 추가
            if "mrc_score" in result:
                processed_result["mrc_score"] = result["mrc_score"]
            if "mrc_answer" in result:
                processed_result["mrc_answer"] = result["mrc_answer"]
            if "mrc_char_ids" in result:
                processed_result["mrc_char_ids"] = result["mrc_char_ids"]
            
            # FlashRank 점수가 있으면 추가
            if "flashrank_score" in result:
                processed_result["flashrank_score"] = result["flashrank_score"]
            
            # 하이브리드 점수가 있으면 추가
            if "hybrid_score" in result:
                processed_result["hybrid_score"] = result["hybrid_score"]
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def hybrid_rerank(
        self, 
        query: str, 
        passages: List[Dict[str, Any]], 
        top_k: Optional[int] = None,
        mrc_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Rerank passages based on query using hybrid approach (FlashRank + MRC)
        
        Args:
            query: Query text
            passages: List of passage dictionaries
            top_k: Number of top results to return
            mrc_weight: Weight for MRC scores (0-1), default is from config
            
        Returns:
            Reranked search results
        """
        payload = {
            "query": query,
            "results": passages,
            "total": len(passages),
            "reranked": False
        }
        
        params = {}
        if top_k is not None:
            params["top_k"] = top_k
        if mrc_weight is not None:
            params["mrc_weight"] = mrc_weight
            
        response = requests.post(
            f"{self.base_url}/hybrid-rerank",
            json=payload,
            params=params
        )
        response.raise_for_status()
        results = response.json()
        
        # 기본 필드만 추출 (중요한 메타데이터 유지)
        processed_results = []
        for result in results:
            processed_result = {
                "id": result.get("id"),  # 고유 식별자 사용
                "doc_id": result["meta"].get("doc_id", ""),
                "passage_id": result["meta"].get("passage_id", ""),
                "text": result["text"],
                "score": float(result["score"])
            }

            # 메타데이터에서 original_score 보존
            if "original_score" in result["meta"]:
                metadata = {
                    "original_score": result["meta"]["original_score"],
                    "unique_id": result.get("id")  # 고유 식별자도 메타데이터에 보존
                }
                processed_result["metadata"] = metadata
            
            # MRC 관련 필드가 있으면 추가
            if "mrc_score" in result:
                processed_result["mrc_score"] = result["mrc_score"]
            if "mrc_answer" in result:
                processed_result["mrc_answer"] = result["mrc_answer"]
            if "mrc_char_ids" in result:
                processed_result["mrc_char_ids"] = result["mrc_char_ids"]
            
            # FlashRank 점수가 있으면 추가
            if "flashrank_score" in result:
                processed_result["flashrank_score"] = result["flashrank_score"]
            
            # 하이브리드 점수가 있으면 추가
            if "hybrid_score" in result:
                processed_result["hybrid_score"] = result["hybrid_score"]
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def batch_rerank(
        self, 
        queries: List[str], 
        passages: List[List[Dict[str, Any]]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch rerank multiple queries with their respective passages
        
        Args:
            queries: List of queries
            passages: List of passage lists corresponding to each query
            top_k: Number of top results to return for each query
            
        Returns:
            List of reranked results for each query
        """
        payload = {
            "queries": queries,
            "passages": passages
        }
        
        params = {}
        if top_k is not None:
            params["top_k"] = top_k
            
        response = requests.post(
            f"{self.base_url}/batch_rerank",
            json=payload,
            params=params
        )
        response.raise_for_status()
        results = response.json()
        
        # 기본 필드만 추출 (중요한 메타데이터 유지)
        processed_results = []
        for query, results in zip(queries, results):
            processed_results.append({
                "query": query,
                "results": [
                    {
                        "id": result.get("id"),  # 고유 식별자 사용
                        "doc_id": result["meta"].get("doc_id", ""),
                        "passage_id": result["meta"].get("passage_id", ""),
                        "text": result["text"],
                        "score": float(result["score"])
                    }
                    for result in results
                ]
            })
        
        return processed_results


# 사용 예시
if __name__ == "__main__":
    # 클라이언트 초기화
    client = RerankerClient("http://localhost:8000")
    
    # 건강 상태 확인
    health = client.health_check()
    print(f"Reranker health: {health}")
    
    # 단일 쿼리 재순위 처리
    query = "메타버스란 무엇인가?"
    passages = [
        {
            "passage_id": 1,
            "doc_id": "doc123",
            "text": "메타버스는 가상 세계를 의미합니다.",
            "score": 0.85
        },
        {
            "passage_id": 2,
            "doc_id": "doc456",
            "text": "메타버스 기술의 발전은 VR 기기의 보급과 함께 가속화되고 있습니다.",
            "score": 0.78
        }
    ]
    
    # FlashRank 기반 재랭킹
    results = client.rerank(query, passages, top_k=5, rerank_type="flashrank")
    print(f"FlashRank results: {json.dumps(results, indent=2, ensure_ascii=False)}")
    
    # MRC 기반 재랭킹
    mrc_results = client.mrc_rerank(query, passages, top_k=5)
    print(f"MRC results: {json.dumps(mrc_results, indent=2, ensure_ascii=False)}")
    
    # 하이브리드 재랭킹
    hybrid_results = client.hybrid_rerank(query, passages, top_k=5, mrc_weight=0.7)
    print(f"Hybrid results: {json.dumps(hybrid_results, indent=2, ensure_ascii=False)}") 