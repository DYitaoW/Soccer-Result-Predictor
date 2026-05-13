import app
from pprint import pprint

with app.app.test_client() as c:
    resp = c.get('/api/h2h', query_string={'team1': 'Liverpool', 'team2': 'Arsenal', 'mode': 'global'})
    print('h2h status', resp.status_code)
    pprint(resp.get_json())

    resp2 = c.get('/api/league-tables', query_string={'mode': 'global'})
    print('league tables status', resp2.status_code)
    data = resp2.get_json()
    print('leagues', data.get('leagues', [])[:5])
    if data.get('leagues'):
        league = data['leagues'][0]
        rows = data['tables'][league]
        print('sample position_odds', rows[0].get('position_odds') if rows else None)
