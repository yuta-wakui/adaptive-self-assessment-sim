import sys, os
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # プロジェクトルート
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)