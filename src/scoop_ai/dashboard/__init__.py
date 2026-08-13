"""Local browser dashboard for operators and reviewers."""

from .app import DashboardServer, run_dashboard, run_product
from .product import ProductManager, ProductPaths

__all__ = ["DashboardServer", "ProductManager", "ProductPaths", "run_dashboard", "run_product"]
