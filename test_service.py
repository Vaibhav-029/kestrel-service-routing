"""
Functional and latency tests for the Kestrel routing service.

Usage:
  1. Start the server:   python -m uvicorn app:app --port 8000
  2. Run tests:          python test_service.py
"""

import json
import time
import requests

BASE = 'http://127.0.0.1:8000'

CANONICAL_TEAMS = {
    'Billing',
    'Filters & Consumables',
    'Installs & Demo',
    'Product Advice',
    'Repairs',
    'Returns & Replacement',
    'Warranty Claims',
}

# ── Functional test cases ────────────────────────────────────────────
FUNCTIONAL_CASES = [
    {
        'name': 'Repair — appliance not turning on',
        'payload': {
            'request_id': 'TEST001',
            'created_at_ist': '2026-06-15T10:30',
            'channel': 'chat',
            'product_family': 'Water Purifier',
            'warranty_status': 'in_warranty',
            'request_text': 'My water purifier is not turning on. The power LED is completely dead since yesterday.',
            'source': 'crm',
        },
    },
    {
        'name': 'Billing — double charge',
        'payload': {
            'request_id': 'TEST002',
            'created_at_ist': '2026-06-15T11:00',
            'channel': 'email',
            'product_family': 'Air Fryer',
            'warranty_status': 'in_warranty',
            'request_text': 'I was charged twice for the same order KO291823. Please issue a refund for the duplicate payment.',
            'source': 'crm',
        },
    },
    {
        'name': 'Installation — new product',
        'payload': {
            'request_id': 'TEST003',
            'created_at_ist': '2026-06-15T12:00',
            'channel': 'ivr',
            'product_family': 'Water Purifier',
            'warranty_status': 'in_warranty',
            'request_text': 'I received my water purifier yesterday but it has not been installed yet. Please schedule an installation visit.',
            'source': 'crm',
        },
    },
    {
        'name': 'Returns — damaged delivery',
        'payload': {
            'request_id': 'TEST004',
            'created_at_ist': '2026-06-15T14:00',
            'channel': 'whatsapp',
            'product_family': 'Mixer Grinder',
            'warranty_status': 'in_warranty',
            'request_text': 'The box arrived damaged and the jar is cracked. I want to return this and get a replacement.',
            'source': 'crm',
        },
    },
    {
        'name': 'Spares — filter replacement',
        'payload': {
            'request_id': 'TEST005',
            'created_at_ist': '2026-06-15T15:00',
            'channel': 'chat',
            'product_family': 'Water Purifier',
            'warranty_status': 'out_of_warranty',
            'request_text': 'I need to replace the RO membrane and sediment filter. How do I order the spare parts?',
            'source': 'crm',
        },
    },
    {
        'name': 'Warranty — shield status inquiry',
        'payload': {
            'request_id': 'TEST006',
            'created_at_ist': '2026-06-15T16:00',
            'channel': 'ivr',
            'product_family': 'Induction Cooktop',
            'warranty_status': 'shield',
            'request_text': 'I want to check my Kestrel Shield warranty claim status and verify coverage for accidental glass damage.',
            'source': 'crm',
        },
    },
    {
        'name': 'Advice — settings and inverter compatibility',
        'payload': {
            'request_id': 'TEST007',
            'created_at_ist': '2026-06-15T17:00',
            'channel': 'chat',
            'product_family': 'Air Fryer',
            'warranty_status': 'in_warranty',
            'request_text': 'What are the recommended temperature settings and can this appliance safely run on a home inverter? No fault reported.',
            'source': 'crm',
        },
    },
]


def run_functional_tests():
    print('=' * 60)
    print('FUNCTIONAL TESTS')
    print('=' * 60)
    passed = 0

    for tc in FUNCTIONAL_CASES:
        resp = requests.post(f'{BASE}/predict', json=tc['payload'], timeout=5)
        ok = resp.status_code == 200
        data = resp.json()

        team_valid = data.get('predicted_team') in CANONICAL_TEAMS
        conf_valid = data.get('confidence') in ('High', 'Medium', 'Low')
        reasons_valid = isinstance(data.get('reasons'), list) and len(data['reasons']) >= 2

        all_ok = ok and team_valid and conf_valid and reasons_valid
        status = 'PASS' if all_ok else 'FAIL'

        print(f'\n{status}  {tc["name"]}')
        print(f'  Team:       {data.get("predicted_team")}')
        print(f'  Confidence: {data.get("confidence")}')
        print(f'  Reasons:    {len(data.get("reasons", []))} items')
        if not team_valid:
            print(f'  !! predicted_team not in canonical set')
        if not conf_valid:
            print(f'  !! confidence not one of High/Medium/Low')
        if not reasons_valid:
            print(f'  !! expected at least 2 reasons')

        if all_ok:
            passed += 1

    print(f'\n{passed}/{len(FUNCTIONAL_CASES)} functional tests passed.\n')
    return passed == len(FUNCTIONAL_CASES)


def run_latency_test(n=200):
    print('=' * 60)
    print(f'LATENCY TEST  ({n} sequential requests)')
    print('=' * 60)

    payload = FUNCTIONAL_CASES[0]['payload']
    latencies = []

    for _ in range(n):
        start = time.perf_counter()
        resp = requests.post(f'{BASE}/predict', json=payload, timeout=5)
        elapsed = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        latencies.append(elapsed)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = sum(latencies) / len(latencies)

    print(f'  Mean:  {mean:.1f} ms')
    print(f'  p50:   {p50:.1f} ms')
    print(f'  p95:   {p95:.1f} ms')
    print(f'  p99:   {p99:.1f} ms')
    print(f'  Min:   {min(latencies):.1f} ms')
    print(f'  Max:   {max(latencies):.1f} ms')
    print()
    return True


def main():
    print(f'Testing service at {BASE}\n')

    # Quick health check
    try:
        r = requests.get(BASE, timeout=5)
        assert r.status_code == 200
        print('Health check: OK\n')
    except Exception as e:
        print(f'Cannot reach service: {e}')
        print('Make sure the server is running: python -m uvicorn app:app --port 8000')
        return

    func_ok = run_functional_tests()
    lat_ok = run_latency_test(200)

    if func_ok and lat_ok:
        print('ALL TESTS PASSED')
    else:
        print('SOME TESTS FAILED')


if __name__ == '__main__':
    main()
