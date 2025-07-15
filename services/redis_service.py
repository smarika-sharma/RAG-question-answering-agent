import json
import hashlib
import time
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
import os
import redis

load_dotenv()

class RedisService:
    def __init__(self):
        # Redis configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_db = int(os.getenv("REDIS_DB", 0))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        
        # TTL settings
        self.cache_ttl = 3600  # 1 hour for cached responses
        self.conversation_ttl = 86400  # 24 hours for conversation history
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            print("Connected to Redis successfully")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            print("Falling back to in-memory storage")
            self.redis_client = None
            self._fallback_cache = {}
            self._fallback_conversations = {}
    
    def _hash_query(self, query: str) -> str:
        """Create a hash for the query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_cache_key(self, query_hash: str) -> str:
        """Get Redis key for cache"""
        return f"cache:{query_hash}"
    
    def _get_conversation_key(self, session_id: str) -> str:
        """Get Redis key for conversation"""
        return f"conversation:{session_id}"
    
    def cache_response(self, query: str, response: str, similarity_score: float = 0.0) -> None:
        """Cache a query-response pair with TTL"""
        try:
            if self.redis_client:
                query_hash = self._hash_query(query)
                cache_key = self._get_cache_key(query_hash)

                cache_data = {
                    "query": query,
                    "response": response,
                    "similarity_score": similarity_score,
                    "timestamp": time.time()
                }
    
                # Store with TTL
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(cache_data)
                )
                
            else:
                # Fallback to in-memory
                query_hash = self._hash_query(query)
                self._fallback_cache[query_hash] = {
                    "query": query,
                    "response": response,
                    "similarity_score": similarity_score,
                    "timestamp": time.time()
                }
        except Exception as e:
            print(f"Cache error: {e}")
    
    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response for a query"""
        try:
            if self.redis_client:
                query_hash = self._hash_query(query)
                cache_key = self._get_cache_key(query_hash)
            
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(str(cached_data))
                return None
            else:
                # Fallback to in-memory
                query_hash = self._hash_query(query)
                return self._fallback_cache.get(query_hash)
        except Exception as e:
            print(f"Get cache error: {e}")
            return None
    
    def find_similar_cached_queries(self, query: str, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar cached queries using simple keyword matching"""
        try:
            if self.redis_client:
                query_words = set(query.lower().split())
                similar_queries = []
                
                # Get all cache keys
                cache_keys = self.redis_client.keys("cache:*")
                
                for key in list(cache_keys):  # type: ignore
                    cached_data = self.redis_client.get(key)
                    if cached_data:
                        data = json.loads(str(cached_data))
                        cached_query_words = set(data["query"].lower().split())
                        intersection = len(query_words.intersection(cached_query_words))
                        union = len(query_words.union(cached_query_words))
                        similarity = intersection / union if union > 0 else 0
                        
                        if similarity >= threshold:
                            data_copy = data.copy()
                            data_copy["similarity"] = similarity
                            similar_queries.append(data_copy)
                
                # Sort by similarity score
                similar_queries.sort(key=lambda x: x["similarity"], reverse=True)
                return similar_queries
            else:
                # Fallback to in-memory
                query_words = set(query.lower().split())
                similar_queries = []
                
                for data in self._fallback_cache.values():
                    cached_query_words = set(data["query"].lower().split())
                    intersection = len(query_words.intersection(cached_query_words))
                    union = len(query_words.union(cached_query_words))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity >= threshold:
                        data_copy = data.copy()
                        data_copy["similarity"] = similarity
                        similar_queries.append(data_copy)
                
                similar_queries.sort(key=lambda x: x["similarity"], reverse=True)
                return similar_queries
        except Exception as e:
            print(f"Similarity search error: {e}")
            return []
    
    def store_conversation(self, session_id: str, user_query: str, response: str, agent_name: str = "unknown") -> None:
        """Store conversation in session memory with TTL"""
        try:
            if self.redis_client:
                conversation_key = self._get_conversation_key(session_id)
    
                # get existing conversation
                existing_data = self.redis_client.get(conversation_key)
                if existing_data:
                    conversation = json.loads(str(existing_data))
                else:
                    conversation = {"messages": []}
                
                # Add new message
                conversation["messages"].append({
                    "timestamp": time.time(),
                    "user_query": user_query,
                    "response": response,
                    "agent_name": agent_name
                })
                
                # Keep only last 20 messages
                if len(conversation["messages"]) > 20:
                    conversation["messages"] = conversation["messages"][-20:]
                
                # Store with TTL
                self.redis_client.setex(
                    conversation_key,
                    self.conversation_ttl,
                    json.dumps(conversation)
                )
            else:
                # Fallback to in-memory
                if session_id not in self._fallback_conversations:
                    self._fallback_conversations[session_id] = {"messages": []}
                
                self._fallback_conversations[session_id]["messages"].append({
                    "timestamp": time.time(),
                    "user_query": user_query,
                    "response": response,
                    "agent_name": agent_name
                })
                
                # Keep only last 20 messages
                if len(self._fallback_conversations[session_id]["messages"]) > 20:
                    self._fallback_conversations[session_id]["messages"] = self._fallback_conversations[session_id]["messages"][-20:]
        except Exception as e:
            print(f"Conversation storage error: {e}")
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        try:
            if self.redis_client:
                conversation_key = self._get_conversation_key(session_id)
                conversation_data = self.redis_client.get(conversation_key)
                
                if conversation_data:
                    conversation = json.loads(str(conversation_data))
                    return conversation.get("messages", [])
                return []
            else:
                # Fallback to in-memory
                conversation = self._fallback_conversations.get(session_id, {"messages": []})
                return conversation.get("messages", [])  # type: ignore
        except Exception as e:
            print(f"Conversation retrieval error: {e}")
            return []
    
    def get_conversation_context(self, session_id: str, max_messages: int = 5) -> str:
        """Get conversation context as a string for the agent"""
        try:
            messages = self.get_conversation_history(session_id)
            
            if not messages:
                return ""
            
            # Get last N messages
            recent_messages = messages[-max_messages:]
            
            context = "Previous conversation:\n"
            for msg in recent_messages:
                context += f"User: {msg['user_query']}\n"
                context += f"Assistant: {msg['response']}\n\n"
            
            return context.strip()
        except Exception as e:
            print(f"Context error: {e}")
            return ""

# Global service instance
redis_service = RedisService() 