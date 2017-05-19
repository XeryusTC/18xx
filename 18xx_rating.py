#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from pprint import pprint
import yaml
import statistics
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from collections import namedtuple

ELO_K = 32
OLD_F1 = [10, 8, 6, 5, 4, 3, 2, 1]
NEW_F1 = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
GLICKO_FACTOR = 173.7178
GLICKO_PERIOD = date(2017, 5, 1) - date(2017, 4, 1)
WIN  = 1
DRAW = 0.5
LOSS = 0

PlayerScore = namedtuple('PlayerScore', ('name', 'score'))

class Player:
    tau = 0.5
    epsilon = 0.00001

    def __init__(self, name):
        self.name = name.title()
        self.scores = []
        self.played = 0
        self.wins = 0
        self.elo = 1000
        self.elo_history = []
        self.old_F1_points = 0
        self.new_F1_points = 0
        self.glicko_mu = 0
        self.glicko_mu_hist = []
        self.glicko_phi = 350/GLICKO_FACTOR
        self.glicko_phi_hist = []
        self.glicko_sigma = 0.06

    def calculate_new_elo(self, other_elo, scores):
        expected = 0
        for other in range(len(other_elo)):
            Q = 10 ** ((other_elo[other] - self.elo) / 400)
            expected += 1 / (1 + Q)
        K = ELO_K
        if self.played < 20:
            K *= self.played / 20
        self._new_elo = self.elo + K * (sum(scores) - expected)

    def update_elo(self, game_num):
        self.elo = self._new_elo
        del self._new_elo
        self.elo_history.append((game_num, self.elo))

    def calculate_new_glicko(self, other_glicko_mu, other_glicko_phi,
                             other_scores):
        pass

    def update_glicko(self, game_num):
        self.glicko_mu = self._new_glicko_mu
        self.glicko_phi = self._new_glicko_phi
        self.glicko_sigma = self._new_glicko_sigma
        del self._new_glicko_mu, self._new_glicko_phi, self._new_glicko_sigma


class Game:
    def __init__(self, name):
        self.name = name
        self.times_played = 0


class Play:
    def __init__(self, game, date, players):
        self.game = game
        self.date = date
        self.ranking = tuple(PlayerScore(p[0], p[1])
            for p in sorted(players.items(), key=lambda p: p[1], reverse=True))


def main():
    with open('results.yml') as f:
        data = yaml.load(f)

    players = {}
    dates = []
    games = {}
    plays = []
    total_games = len(data)

    for game in reversed(data):
        print(f"Processing game from {game['date']}")
        dates.append(game['date'])
        # Register a play of this game
        if game['game'] not in games:
            games[game['game']] = Game(game['game'])
        games[game['game']].times_played += 1

        # Record the information of the play
        play = Play(games[game['game']], game['date'], game['players'])
        plays.append(play)
        game_num = len(plays)

        # Record information for all players
        for name, score in game['players'].items():
            name = name.lower()
            if name not in players:
                players[name] = Player(name)
            players[name].scores.append(score)
            players[name].played += 1

        print('  Rank of players:', ', '.join((p.name.title()
                                               for p in play.ranking)))

        # Increase number of wins for the winning player
        players[play.ranking[0].name].wins += 1

        for player in range(len(play.ranking)):
            scores = []
            elos = []
            glicko_mus = []
            glicko_phis = []
            for other in range(len(play.ranking)):
                if other == player:
                    continue
                elos.append(players[play.ranking[other].name].elo)
                glicko_mus.append(players[play.ranking[other].name].glicko_mu)
                glicko_phis.append(
                    players[play.ranking[other].name].glicko_phi)

                # Determine win/loss against each opponent
                if play.ranking[player].score > play.ranking[other].score:
                    scores.append(WIN)
                elif play.ranking[player].score < play.ranking[other].score:
                    scores.append(LOSS)
                else:
                    scores.append(DRAW)

            # Calculate new score (but don't update)
            players[play.ranking[player].name].calculate_new_elo(elos, scores)
            players[play.ranking[player].name].calculate_new_glicko(
                glicko_mus, glicko_phis, scores)
        for player in play.ranking:
            players[player.name].update_elo(game_num)
            #players[player.name].update_glicko(game_num)
            print("  {} new ELO: {:.0f}".format(player.name.title(),
                                                players[player.name].elo))

        # Distribute F1 points
        for i in range(len(play.ranking)):
            players[play.ranking[i].name].old_F1_points += OLD_F1[i]
            players[play.ranking[i].name].new_F1_points += NEW_F1[i]

        # Calculate glicko scores
        #calculate_glicko(players, ranked, game['date'], game_num)
        # Update last played game
        #players[name]['last_game'] = game['date']

    html_results('rankings.html', players, games, total_games, dates, plays)
    plot_elo(players)
    plot_glicko(players)

def calculate_glicko(players, ranked, game_date, game_num):
    tau = 0.5
    epsilon = 0.0001
    def g(phi):
        return 1 / np.sqrt(1 + (3 * phi * phi) / (np.pi * np.pi))

    def E(mu, other_mu, other_phi):
        return 1 / (1 + np.exp(-g(other_phi) * (mu - other_mu)))

    def sigma_prime_iter(sigma, phi, Delta, v):
        # Step 1
        def f(x):
            n1 = np.e ** x * (Delta ** 2 - phi ** 2 - v - np.e ** x)
            d1 = 2 * (phi ** 2 + v + e ** x) ** 2
            n2 = x - a
            d2 = tau * tau
            return (n1 / d1) - (n2 / d2)
        # Step 2
        A = a = np.log(sigma * sigma)
        if Delta ** 2 > phi ** 2 + v:
            B = np.log(Delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau
        # Step 3
        fA = f(A)
        fB = f(B)
        # Step 4
        while abs(B - A) > epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA /= 2
            B = C
            fB = fC
        # Step 5
        return np.e ** (A / 2)

    phi_primes = {}
    mu_primes = {}
    sigma_primes = {}
    for player in ranked:
        mu = players[player]['glicko_mu']
        # Calculate v and Delta
        v = 0
        delta = 0
        for other in ranked:
            # Dont calculate glicko against self
            if player == other:
                continue
            # Get the score
            if players[player]['scores'][-1] > players[other]['scores'][-1]:
                score = 1
            elif players[player]['scores'][-1] == players[other]['scores'][-1]:
                score = 0.5
            else:
                score = 0
            other_mu  = players[other]['glicko_mu']
            other_phi = players[other]['glicko_phi']
            e = E(mu, other_mu, other_phi)
            v += g(other_phi) ** 2 * e * (1 - e)
            delta += g(other_phi) * (score - e)
        v = 1 / v
        Delta = v * delta

        # Calculate the actual ratings
        phi = players[player]['glicko_phi']
        sigma_prime = sigma_prime_iter(players[player]['glicko_sigma'],
                                       phi, Delta, v)
        phi_star = np.sqrt(phi * phi + sigma_prime * sigma_prime)
        phi_prime = 1 / np.sqrt((1 / (phi_star * phi_star)) + 1 / v)
        mu_prime = mu + phi_prime * phi_prime * delta
        # Store new ratings for later updates
        phi_primes[player]   = phi_prime
        mu_primes[player]    = mu_prime
        sigma_primes[player] = sigma_prime

    # Update the ratings
    for player in ranked:
        players[player]['glicko_mu'] = mu_primes[player]
        players[player]['glicko_phi'] = phi_primes[player]
        players[player]['glicko_sigma'] = sigma_primes[player]
        # Update the history
        players[player]['glicko_mu_hist'].append((game_num, mu_primes[player]))
        players[player]['glicko_phi_hist'].append((game_num,
                                                   phi_primes[player]))
        players[player]['glicko_sigma_hist'].append((game_num,
                                                     sigma_primes[player]))
        print(f'  {player.title()} new glicko:',
              f'{mu_primes[player] * GLICKO_FACTOR + 1500:.2f} +=',
              f'{phi_primes[player] * GLICKO_FACTOR:.2f}')
    # Increase RD for unranked players
    for player in players:
        # Skip just calculated players
        if player in ranked:
            continue
        if game_date - players[player]['last_game'] > GLICKO_PERIOD:
            phi = players[player]['glicko_phi']
            sigma = players[player]['glicko_sigma']
            phi_prime = np.sqrt(phi * phi + sigma * sigma)
            players[player]['glicko_phi'] = phi_prime
            # Update the player's history
            players[player]['glicko_mu_hist'].append(
                (game_num, players[player]['glicko_mu']))
            players[player]['glicko_phi_hist'].append((game_num, phi_prime))
            players[player]['glicko_sigma_hist'].append((game_num, sigma))
            print(f'  {player.title()} glicko RD updated due to inactivity')

def plot_elo(players):
    plt.clf()
    labels = []
    for player in players:
        elo = np.array(players[player].elo_history).T
        line, = plt.plot(*elo, 'x-', label=player.title())
        labels.append(line)
    plt.title('History of ELO ratings')
    plt.xlabel('Games played')
    plt.ylabel('ELO rating')
    plt.legend(handles=labels)
    plt.savefig('rankings_elo.png')

def plot_glicko(players):
    plt.clf()
    labels = []
    ax = plt.gca() # Get current axes
    for player in players:
        # plot
        glicko_mu = np.array(players[player].glicko_mu_hist).T
        glicko_mu[1] = glicko_mu[1] * GLICKO_FACTOR + 1500
        glicko_phi = np.array(players[player].glicko_phi_hist).T[1]
        glicko_phi *= GLICKO_FACTOR * 2
        line, = plt.plot(*glicko_mu, label=player.title())
        labels.append(line)

        # Plot confidence interval
        low = glicko_mu.copy()
        high = glicko_mu.copy()
        low[1] -= 2 * glicko_phi
        high[1] += 2 * glicko_phi
        plt.plot(*low, '-', color=line.get_color(), alpha=0.5,
                 linewidth=line.get_linewidth()*.5)
        plt.plot(*high, '-', color=line.get_color(), alpha=0.5,
                 linewidth=line.get_linewidth()*.5)
        poly_coords = np.concatenate([low.T, np.flipud(high.T)])
        poly = mpatches.Polygon(poly_coords,
                                facecolor=line.get_color(),
                                alpha=0.1)
        ax.add_patch(poly)

    ax.set_ylim(1000, 2000)
    plt.title('History of Glicko ratings')
    plt.xlabel('Games played')
    plt.ylabel('Glicko rating')
    plt.legend(handles=labels, loc=2)
    plt.savefig('rankings_glicko.png')

def html_results(filename, players, games, total_games, dates, plays):
    dates.sort()
    with open(filename, 'w') as f:
        f.write(f'''<!doctype html>
        <html>
        <head><title>18xx rankings</title></head>
        <body>
        <h1>Global stats</h1>
        Total games played: {total_games}<br />
        First recorded game: {dates[0]}<br />
        Last recorded game: {dates[-1]}<br />

        <h1>Player stats</h1>
        <table border="1">
        <tr>
            <th>Name</th>
            <th>Played</th>
            <th>Wins</th>
            <th>Ratio</th>
            <th>ELO</th>
            <th>Glicko rating</th>
            <th>Glicko RD</th>
            <th>Glicko 95% CI</th>
            <th>Old F1 points</th>
            <th>New F1 points</th>
            <th>Total score</th>
            <th>Mean score</th>
            <th>Median score</th>
        </tr>''')

        for name, player in sorted(players.items()):
            total_score = sum(player.scores)
            mean_score = total_score / player.played
            mean_old_F1 = player.old_F1_points / player.played
            mean_new_F1 = player.new_F1_points / player.played
            ci_lo       = player.glicko_mu - 2 * player.glicko_phi
            ci_hi       = player.glicko_mu + 2 * player.glicko_phi
            f.write(f'''<tr style="text-align:right;">
                <td>{name.title()}</td>
                <td>{player.played}</td>
                <td>{player.wins}</td>
                <td>{player.wins / player.played:.2f}</td>
                <td>{player.elo:.0f}</td>
                <td>{player.glicko_mu * GLICKO_FACTOR + 1500:.2f}</td>
                <td>{player.glicko_phi * GLICKO_FACTOR:.2f}</td>
                <td>
                    {ci_lo * GLICKO_FACTOR + 1500:.2f}-{ci_hi * GLICKO_FACTOR + 1500:.2f}
                </td>
                <td>{player.old_F1_points} ({mean_old_F1:.2f})</td>
                <td>{player.new_F1_points} ({mean_new_F1:.2f})</td>
                <td>{sum(player.scores)}</td>
                <td>{statistics.mean(player.scores):.2f}</td>
                <td>{statistics.median(player.scores):.0f}</td>
            </tr>''')

        f.write('''</table><br />
        <img src="rankings_elo.png" />
        <img src="rankings_glicko.png" />
        <table border="1">
            <tr><td></td>''')
        for i in range(len(NEW_F1)):
            f.write(f"<th>{i+1}</th>")
        f.write('</tr><tr><th>Old F1</th>')
        for score in OLD_F1:
            f.write(f"<td>{score}</td>")
        f.write('</tr><tr><th>New F1</th>')
        for score in NEW_F1:
            f.write(f"<td>{score}</td>")

        f.write('''</tr></table>
        <h1>Game stats</h1>
        <table border="1">
        <tr>
            <th>Name</th>
            <th>Played</th>
        </tr>''')
        for name, game in sorted(games.items()):
            f.write(f'''<tr>
                <td>{game.name}</td>
                <td>{game.times_played}</td>
            </tr>''')
        f.write('''</table>
            <h1>Recorded games</h1>
            <table border="1">
            <tr>
                <th>Date</th>
                <th>Game</th>
                <th>Player</th>
                <th>Score</th>
            </tr>''')
        for play in reversed(plays):
            f.write(f'''<tr>
                <td rowspan="{len(play.ranking)}">{play.date}</td>
                <td rowspan="{len(play.ranking)}">{play.game.name}</td>
                ''')
            first = True
            for player in play.ranking:
                if not first:
                    f.write('<tr>')
                f.write(f'''<td>{player.name.title()}</td>
                    <td>{player.score}</td>
                </tr>''')

        f.write('</table></body></html>\n')

if __name__ == '__main__':
    main()
