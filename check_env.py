import importlib

# 你要用到的主要包
required = {
    "torch": "2.8",
    "transformers": "4.56",
    "datasets": "4.1",
    "tqdm": "4.67",
    "numpy": "2.3",
    "pandas": "2.3",
    "yaml": "6.0",
    "tokenizers": "0.22",
}

print("="*60)
print("📦 Required packages and expected versions")
for pkg, ver in required.items():
    print(f" - {pkg:<12}  (expected ~{ver})")
print("="*60)
print("🔍 Checking Python environment...")
print("="*60)

for pkg, expected in required.items():
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"✅ {pkg:<12} installed, version {version} (expected ~{expected})")
    except ImportError:
        print(f"❌ {pkg:<12} NOT installed!")

print("="*60)
print("Check complete.")
print("如果有 ❌ ，请重新安装对应的包： pip install 包名==版本")
