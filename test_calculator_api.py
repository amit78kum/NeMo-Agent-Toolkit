#!/usr/bin/env python3
"""
Test script for the NeMo Agent Toolkit Calculator API
"""

import requests
import json
import sys

# Configuration
API_URL = "http://localhost:8000/generate"
HEADERS = {"Content-Type": "application/json"}

# Test cases
TEST_CASES = [
    {
        "name": "Simple Addition",
        "input_message": "What is 2 + 3?"
    },
    {
        "name": "Comparison with Time",
        "input_message": "Is 40 + 40 greater than the current hour of the day?"
    },
    {
        "name": "Complex Math",
        "input_message": "Calculate 10 * 5 + 3"
    },
    {
        "name": "Multiple Operations",
        "input_message": "What is (20 - 5) * 2?"
    }
]

def test_api():
    """Test the calculator API endpoint"""
    print("=" * 60)
    print("Testing NeMo Agent Toolkit Calculator API")
    print("=" * 60)
    print(f"API URL: {API_URL}\n")
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 60)
        print(f"Input: {test_case['input_message']}")
        
        try:
            payload = {"input_message": test_case['input_message']}
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=30
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print("Response:")
                    print(json.dumps(result, indent=2))
                    results.append({"test": test_case['name'], "status": "PASS", "response": result})
                except json.JSONDecodeError:
                    print(f"Response (raw): {response.text}")
                    results.append({"test": test_case['name'], "status": "FAIL", "error": "Invalid JSON response"})
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                results.append({"test": test_case['name'], "status": "FAIL", "error": f"HTTP {response.status_code}"})
                
        except requests.exceptions.ConnectionError:
            print("ERROR: Could not connect to API. Make sure the server is running on http://localhost:8000")
            results.append({"test": test_case['name'], "status": "FAIL", "error": "Connection refused"})
        except requests.exceptions.Timeout:
            print("ERROR: Request timed out")
            results.append({"test": test_case['name'], "status": "FAIL", "error": "Request timeout"})
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({"test": test_case['name'], "status": "FAIL", "error": str(e)})
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    
    for result in results:
        status_icon = "✓" if result["status"] == "PASS" else "✗"
        print(f"{status_icon} {result['test']}: {result['status']}")
        if result["status"] == "FAIL":
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "-" * 60)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
