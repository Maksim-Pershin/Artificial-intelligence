from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseSkill(ABC):
    """Базовый класс для всех навыков"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, text: str, bot_instance=None, **kwargs) -> str:
        """Выполнение навыка"""
        pass
    
    def get_description(self) -> str:
        return f"{self.name}: {self.description}"