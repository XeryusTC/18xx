#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from pprint import pprint
import yaml
import statistics

elo_K = 32
old_F1 = [10, 8, 6, 5, 4, 3, 2, 1]
new_F1 = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

def main():
    with open('results.yml') as f:
        data = yaml.load(f)

    players = {}
    dates = []
    games = {}
    total_games = len(data)

    for game in reversed(data):
        print(f"Processing game {game['date']}")
        dates.append(game['date'])
        if game['game'] not in games:
            games[game['game']] = 0
        games[game['game']] += 1
        ranked = []
        for name, score in game['players'].items():
            name = name.lower()
            if name not in players:
                players[name] = {
                    'scores': [],
                    'played': 0,
                    'elo': 1000,
                    'old_points': 0,
                    'new_points': 0,
                    'wins': 0,
                }
            players[name]['scores'].append(score)
            players[name]['played'] += 1

            # Find place in ranking
            if ranked == []:
                ranked = [name]
            else:
                for r in range(len(ranked)):
                    if players[ranked[r]]['scores'][-1] < score:
                        ranked.insert(r, name)
                        break
                else:
                    ranked.append(name)
        print('  Rank of players:', ', '.join(ranked))

        # Increase number of wins
        players[ranked[0]]['wins'] += 1

        # Calculate ELO score
        expected = {}
        scored = {}
        other_elo = {}
        for name in ranked:
            scored[name] = len(ranked) - ranked.index(name) - 1
            expected[name] = 0
            other_elo[name] = 0
            # Calculate expected score
            for other in ranked:
                # Don't compare against self
                if name == other:
                    continue
                # If players have same score, make it a draw
                if players[name]['elo'] == players[other]['elo']:
                    if ranked.index(name) > ranked.index(other):
                        scored[name] -= .5
                    else:
                        scored[name] += .5
                # Update expected score
                Q = 10**((players[other]['elo'] - players[name]['elo']) / 400)
                expected[name] += 1 / (1 + Q)
                other_elo[name] += players[other]['elo']
        # Update ELO scores
        for name in ranked:
            players[name]['elo'] += elo_K * (scored[name] - expected[name])
            print(f"  {name.title()} new ELO: {players[name]['elo']:.0f}")

        # Distribute F1 points
        for i in range(len(ranked)):
            players[ranked[i]]['old_points'] += old_F1[i]
            players[ranked[i]]['new_points'] += new_F1[i]

    html_results(players, 'rankings.html')

def html_results(players, filename):
    with open(filename, 'w') as f:
        f.write('''<!doctype html>
        <html>
        <head><title>18xx rankings</title></head>
        <body>
        <table border="1">
            <tr><td></td>''')
        for i in range(len(new_F1)):
            f.write(f"<th>{i+1}</th>")
        f.write('</tr><tr><th>Old F1</th>')
        for score in old_F1:
            f.write(f"<td>{score}</td>")
        f.write('</tr><tr><th>New F1</th>')
        for score in new_F1:
            f.write(f"<td>{score}</td>")
        f.write('''</tr>
        </table><br />

        <table border="1">
        <tr>
            <th>Name</th>
            <th>Played</th>
            <th>Wins</th>
            <th>Ratio</th>
            <th>ELO</th>
            <th>Old F1 points</th>
            <th>New F1 points</th>
            <th>Total score</th>
            <th>Mean score</th>
            <th>Median score</th>
        </tr>''')

        for player, stats in sorted(players.items()):
            total_score = sum(stats['scores'])
            mean_score = total_score / stats['played']
            f.write(f'''<tr style="text-align:right;">
                <td>{player.title()}</td>
                <td>{stats['played']}</td>
                <td>{stats['wins']}</td>
                <td>{stats['wins'] / stats['played']:.2f}</td>
                <td>{stats['elo']:.0f}</td>
                <td>{stats['old_points']}</td>
                <td>{stats['new_points']}</td>
                <td>{total_score}</td>
                <td>{mean_score:.2f}</td>
                <td>{statistics.median(stats['scores']):.0f}</td>
            </tr>''')

        f.write('</table></body></html>\n')

if __name__ == '__main__':
    main()
