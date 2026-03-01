#!/usr/bin/env python3
"""
后端健康检查脚本 - 验证 API 是否正常运行
"""
import requests
import sys

API_BASE = "http://localhost:8000"

print("=" * 60)
print("🔍 Spatial Shift 后端健康检查")
print("=" * 60)

try:
    # 测试根端点
    print("\n1️⃣ 测试根端点 (GET /)...")
    response = requests.get(API_BASE, timeout=5)
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ 成功！服务状态: {data.get('status')}")
        print(f"   📍 可用端点: {', '.join(data.get('endpoints', []))}")
    else:
        print(f"   ❌ 失败！状态码: {response.status_code}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 后端 API 运行正常！")
    print("=" * 60)
    print("\n📝 下一步：")
    print("   1. 打开新终端")
    print("   2. cd frontend")
    print("   3. npm run dev")
    print("   4. 浏览器打开前端地址")
    print("\n✨ 享受使用 Spatial Shift！")
    
except requests.exceptions.ConnectionError:
    print("\n❌ 无法连接到后端！")
    print("\n💡 解决方案：")
    print("   1. 确认后端正在运行:")
    print("      cd backend && python3 main.py")
    print("   2. 检查端口 8000 是否被占用:")
    print("      lsof -i :8000")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    sys.exit(1)
