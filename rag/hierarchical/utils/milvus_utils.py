"""
Milvus 유틸리티

위계형 RAG 시스템에서 Milvus 관련 작업을 위한 헬퍼 함수들을 제공합니다.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
from pymilvus import Collection, utility, connections, DataType


class MilvusHelper:
    """Milvus 작업 헬퍼 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_collection_statistics(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 통계 조회"""
        try:
            if not utility.has_collection(collection_name):
                return {"error": f"컬렉션 {collection_name}이 존재하지 않습니다"}
            
            collection = Collection(collection_name)
            collection.load()
            
            stats = {
                "collection_name": collection_name,
                "total_entities": collection.num_entities,
                "is_empty": collection.is_empty,
                "description": collection.description,
                "schema_info": self._get_schema_info(collection),
                "index_info": self._get_index_info(collection),
                "partition_info": self._get_partition_info(collection),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"컬렉션 통계 조회 중 오류: {e}")
            return {"error": str(e)}
    
    def _get_schema_info(self, collection: Collection) -> Dict[str, Any]:
        """스키마 정보 추출"""
        try:
            schema = collection.schema
            
            fields_info = []
            for field in schema.fields:
                field_info = {
                    "name": field.name,
                    "type": str(field.dtype),
                    "is_primary": field.is_primary,
                    "auto_id": getattr(field, 'auto_id', False),
                    "description": getattr(field, 'description', "")
                }
                
                # 추가 속성들
                if hasattr(field, 'max_length'):
                    field_info["max_length"] = field.max_length
                if hasattr(field, 'dim'):
                    field_info["dimension"] = field.dim
                    
                fields_info.append(field_info)
            
            return {
                "total_fields": len(schema.fields),
                "primary_field": schema.primary_field.name if schema.primary_field else None,
                "enable_dynamic_field": schema.enable_dynamic_field,
                "fields": fields_info
            }
            
        except Exception as e:
            self.logger.error(f"스키마 정보 추출 중 오류: {e}")
            return {"error": str(e)}
    
    def _get_index_info(self, collection: Collection) -> Dict[str, Any]:
        """인덱스 정보 추출"""
        try:
            indexes_info = []
            
            for field in collection.schema.fields:
                try:
                    if field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                        index = collection.index(field.name)
                        if index:
                            indexes_info.append({
                                "field_name": field.name,
                                "index_type": str(index.index_type),
                                "metric_type": str(index.metric_type),
                                "params": index.params
                            })
                except Exception:
                    # 인덱스가 없는 필드는 무시
                    continue
            
            return {
                "total_indexes": len(indexes_info),
                "indexes": indexes_info
            }
            
        except Exception as e:
            self.logger.error(f"인덱스 정보 추출 중 오류: {e}")
            return {"error": str(e)}
    
    def _get_partition_info(self, collection: Collection) -> Dict[str, Any]:
        """파티션 정보 추출"""
        try:
            partitions = collection.partitions
            
            partition_info = []
            for partition in partitions:
                partition_info.append({
                    "name": partition.name,
                    "num_entities": partition.num_entities,
                    "is_empty": partition.is_empty
                })
            
            return {
                "total_partitions": len(partitions),
                "partitions": partition_info
            }
            
        except Exception as e:
            self.logger.error(f"파티션 정보 추출 중 오류: {e}")
            return {"error": str(e)}
    
    def analyze_hierarchy_distribution(self, collection_name: str) -> Dict[str, Any]:
        """위계 분포 분석"""
        try:
            if not utility.has_collection(collection_name):
                return {"error": f"컬렉션 {collection_name}이 존재하지 않습니다"}
            
            collection = Collection(collection_name)
            collection.load()
            
            # 위계 레벨별 분포 조회 (간단한 구현)
            # 실제로는 더 복잡한 쿼리가 필요할 수 있음
            
            hierarchy_stats = {
                "collection_name": collection_name,
                "total_nodes": collection.num_entities,
                "level_distribution": {},  # 실제 구현에서는 적절한 쿼리 필요
                "node_type_distribution": {},  # 실제 구현에서는 적절한 쿼리 필요
                "analysis_note": "상세 분석은 추가 구현 필요"
            }
            
            return hierarchy_stats
            
        except Exception as e:
            self.logger.error(f"위계 분포 분석 중 오류: {e}")
            return {"error": str(e)}
    
    def check_collection_health(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 건강 상태 확인"""
        try:
            health_report = {
                "collection_name": collection_name,
                "status": "unknown",
                "issues": [],
                "recommendations": [],
                "overall_health": "unknown"
            }
            
            # 컬렉션 존재 여부 확인
            if not utility.has_collection(collection_name):
                health_report["status"] = "not_found"
                health_report["issues"].append("컬렉션이 존재하지 않습니다")
                health_report["overall_health"] = "critical"
                return health_report
            
            collection = Collection(collection_name)
            
            # 기본 상태 확인
            try:
                collection.load()
                health_report["status"] = "loaded"
            except Exception as e:
                health_report["status"] = "load_failed"
                health_report["issues"].append(f"컬렉션 로드 실패: {str(e)}")
            
            # 데이터 확인
            if collection.is_empty:
                health_report["issues"].append("컬렉션이 비어있습니다")
            else:
                entity_count = collection.num_entities
                if entity_count < 10:
                    health_report["issues"].append(f"데이터가 매우 적습니다 ({entity_count}개)")
                elif entity_count > 1000000:
                    health_report["recommendations"].append("대용량 데이터로 인한 성능 최적화 검토 필요")
            
            # 인덱스 확인
            has_vector_index = False
            for field in collection.schema.fields:
                if field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                    try:
                        index = collection.index(field.name)
                        if index:
                            has_vector_index = True
                    except Exception:
                        health_report["issues"].append(f"벡터 필드 {field.name}에 인덱스가 없습니다")
            
            if not has_vector_index:
                health_report["issues"].append("벡터 인덱스가 설정되지 않았습니다")
            
            # 전체 건강 상태 판정
            if len(health_report["issues"]) == 0:
                health_report["overall_health"] = "healthy"
            elif len(health_report["issues"]) <= 2:
                health_report["overall_health"] = "warning" 
            else:
                health_report["overall_health"] = "critical"
            
            return health_report
            
        except Exception as e:
            self.logger.error(f"컬렉션 건강 상태 확인 중 오류: {e}")
            return {
                "collection_name": collection_name,
                "status": "error",
                "error": str(e),
                "overall_health": "critical"
            }
    
    def optimize_collection_performance(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 성능 최적화 제안"""
        try:
            optimization_report = {
                "collection_name": collection_name,
                "current_performance": {},
                "optimization_suggestions": [],
                "estimated_improvement": "unknown"
            }
            
            if not utility.has_collection(collection_name):
                return {"error": f"컬렉션 {collection_name}이 존재하지 않습니다"}
            
            collection = Collection(collection_name)
            stats = self.get_collection_statistics(collection_name)
            
            # 성능 분석
            entity_count = stats.get("total_entities", 0)
            
            # 최적화 제안 생성
            suggestions = []
            
            if entity_count > 100000:
                suggestions.append({
                    "category": "indexing",
                    "suggestion": "대용량 데이터를 위한 IVF_PQ 인덱스 사용 고려",
                    "impact": "high",
                    "effort": "medium"
                })
            
            if entity_count > 1000000:
                suggestions.append({
                    "category": "partitioning",
                    "suggestion": "데이터 파티셔닝으로 쿼리 성능 향상",
                    "impact": "high", 
                    "effort": "high"
                })
            
            # 기본 제안들
            suggestions.extend([
                {
                    "category": "caching",
                    "suggestion": "자주 검색되는 쿼리에 대한 캐싱 구현",
                    "impact": "medium",
                    "effort": "low"
                },
                {
                    "category": "batch_processing",
                    "suggestion": "배치 크기 최적화로 처리량 향상",
                    "impact": "medium",
                    "effort": "low"
                }
            ])
            
            optimization_report["optimization_suggestions"] = suggestions
            optimization_report["current_performance"] = {
                "entity_count": entity_count,
                "index_count": len(stats.get("index_info", {}).get("indexes", [])),
                "partition_count": stats.get("partition_info", {}).get("total_partitions", 1)
            }
            
            return optimization_report
            
        except Exception as e:
            self.logger.error(f"성능 최적화 분석 중 오류: {e}")
            return {"error": str(e)}
    
    def backup_collection_schema(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 스키마 백업"""
        try:
            if not utility.has_collection(collection_name):
                return {"error": f"컬렉션 {collection_name}이 존재하지 않습니다"}
            
            collection = Collection(collection_name)
            schema = collection.schema
            
            # 스키마 정보를 직렬화 가능한 형태로 변환
            schema_backup = {
                "collection_name": collection_name,
                "backup_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "schema": {
                    "description": schema.description,
                    "enable_dynamic_field": schema.enable_dynamic_field,
                    "fields": []
                }
            }
            
            for field in schema.fields:
                field_info = {
                    "name": field.name,
                    "dtype": str(field.dtype),
                    "is_primary": field.is_primary,
                    "description": getattr(field, 'description', "")
                }
                
                # 타입별 추가 속성
                if hasattr(field, 'max_length'):
                    field_info["max_length"] = field.max_length
                if hasattr(field, 'dim'):
                    field_info["dim"] = field.dim
                if hasattr(field, 'auto_id'):
                    field_info["auto_id"] = field.auto_id
                    
                schema_backup["schema"]["fields"].append(field_info)
            
            return {
                "success": True,
                "backup": schema_backup,
                "usage_note": "이 백업을 사용하여 동일한 스키마의 컬렉션을 재생성할 수 있습니다"
            }
            
        except Exception as e:
            self.logger.error(f"스키마 백업 중 오류: {e}")
            return {"error": str(e)}
    
    def compare_collections(self, collection1: str, collection2: str) -> Dict[str, Any]:
        """두 컬렉션 비교"""
        try:
            comparison = {
                "collection1": collection1,
                "collection2": collection2,
                "comparison_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "differences": [],
                "similarities": [],
                "recommendations": []
            }
            
            # 두 컬렉션 모두 존재하는지 확인
            if not utility.has_collection(collection1):
                return {"error": f"컬렉션 {collection1}이 존재하지 않습니다"}
            
            if not utility.has_collection(collection2):
                return {"error": f"컬렉션 {collection2}이 존재하지 않습니다"}
            
            # 통계 정보 수집
            stats1 = self.get_collection_statistics(collection1)
            stats2 = self.get_collection_statistics(collection2)
            
            # 엔티티 수 비교
            entities1 = stats1.get("total_entities", 0)
            entities2 = stats2.get("total_entities", 0)
            
            if entities1 != entities2:
                comparison["differences"].append({
                    "aspect": "entity_count",
                    "collection1_value": entities1,
                    "collection2_value": entities2,
                    "difference": abs(entities1 - entities2)
                })
            else:
                comparison["similarities"].append("동일한 엔티티 수")
            
            # 스키마 비교 (간단한 구현)
            fields1 = stats1.get("schema_info", {}).get("total_fields", 0)
            fields2 = stats2.get("schema_info", {}).get("total_fields", 0)
            
            if fields1 != fields2:
                comparison["differences"].append({
                    "aspect": "field_count",
                    "collection1_value": fields1,
                    "collection2_value": fields2,
                    "difference": abs(fields1 - fields2)
                })
            else:
                comparison["similarities"].append("동일한 필드 수")
            
            # 권장사항 생성
            if len(comparison["differences"]) == 0:
                comparison["recommendations"].append("두 컬렉션이 유사합니다. 통합을 고려해보세요.")
            else:
                comparison["recommendations"].append("컬렉션 간 차이점을 분석하여 최적화 방안을 수립하세요.")
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"컬렉션 비교 중 오류: {e}")
            return {"error": str(e)}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Milvus 시스템 정보 조회"""
        try:
            system_info = {
                "connection_status": "unknown",
                "available_collections": [],
                "total_collections": 0,
                "system_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            try:
                # 연결 상태 확인
                collections = utility.list_collections()
                system_info["connection_status"] = "connected"
                system_info["available_collections"] = collections
                system_info["total_collections"] = len(collections)
                
                # 각 컬렉션의 간단한 정보
                collection_summaries = []
                for collection_name in collections[:10]:  # 최대 10개만
                    try:
                        collection = Collection(collection_name)
                        summary = {
                            "name": collection_name,
                            "entity_count": collection.num_entities,
                            "is_empty": collection.is_empty
                        }
                        collection_summaries.append(summary)
                    except Exception:
                        continue
                
                system_info["collection_summaries"] = collection_summaries
                
            except Exception as e:
                system_info["connection_status"] = "disconnected"
                system_info["connection_error"] = str(e)
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"시스템 정보 조회 중 오류: {e}")
            return {"error": str(e)}
