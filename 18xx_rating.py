#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
from datetime import date, timedelta
from collections import namedtuple
from jinja2 import Environment, FileSystemLoader, select_autoescape
import calendar
import sys
from unipath import Path
import math
from shutil import copyfile

ELO_K = 32
GLICKO_FACTOR = 173.7178
GLICKO_PERIOD = date(2017, 5, 1) - date(2017, 4, 1)
WIN  = 1
DRAW = 0.5
LOSS = 0
DRAW_DELTA = 0
OUTDIR = Path('output')

PlayerScore = namedtuple('PlayerScore', ('name', 'score'))


class Player:
    glicko_tau = 0.43
    glicko_epsilon = 0.00001

    def __init__(self, name, start_date):
        self.name = name.title()
        self.scores = []
        self.played = 0
        self.wins = 0
        self.elo = 1000
        self.elo_history = []
        self.glicko_mu = 0
        self.glicko_mu_hist = [(start_date, self.glicko_mu)]
        self.glicko_phi = 350/GLICKO_FACTOR
        self.glicko_phi_hist = [(start_date, self.glicko_phi)]
        self.glicko_sigma = 0.08
        self.glicko_inactive = True

    def calculate_new_elo(self, other_elo, scores):
        expected = 0
        for other in range(len(other_elo)):
            Q = 10 ** ((other_elo[other] - self.elo) / 400)
            expected += 1 / (1 + Q)
        self._new_elo = self.elo + ELO_K * (sum(scores) - expected)

    def update_elo(self, game_num):
        self.elo = self._new_elo
        del self._new_elo
        self.elo_history.append((game_num, self.elo))

    def calculate_new_glicko(self, other_mu, other_phi, scores):
        self.glicko_inactive = False
        mu_prime = 0
        phi_prime = 0
        sigma_prime = 0
        v = 0
        delta = 0
        for other in range(len(other_mu)):
            e = self._glicko_E(self.glicko_mu, other_mu[other],
                               other_phi[other])
            v += np.power(self._glicko_g(other_phi[other]), 2) * e * (1 - e)
            delta += self._glicko_g(other_phi[other]) * (scores[other] - e)
        v = 1 / v
        Delta = v * delta
        sigma_prime = self._glicko_sigma_iter(self.glicko_sigma,
                                              self.glicko_phi, Delta, v)
        # Update the RD (phi)
        phi_star = np.sqrt(np.power(self.glicko_phi, 2) +
                           np.power(sigma_prime, 2))
        phi_prime = 1 / np.sqrt(1 / (phi_star * phi_star) + 1/v)
        mu_prime = self.glicko_mu + np.power(phi_prime, 2) * delta
        # Store new Glicko variables for later update
        self._new_glicko_mu = mu_prime
        self._new_glicko_phi = phi_prime
        self._new_glicko_sigma = sigma_prime

    def calculate_glicko_inactive(self):
        if self.glicko_inactive: # Don't increase glicko if no games
            phi_prime = self.glicko_phi
        else:
            phi_prime = np.sqrt(np.power(self.glicko_phi, 2) +
                                np.power(self.glicko_sigma, 2))
        self._new_glicko_mu = self.glicko_mu
        self._new_glicko_phi = phi_prime
        self._new_glicko_sigma = self.glicko_sigma

    def update_glicko(self, game_num):
        self.glicko_mu = self._new_glicko_mu
        self.glicko_phi = self._new_glicko_phi
        self.glicko_sigma = self._new_glicko_sigma
        del self._new_glicko_mu, self._new_glicko_phi, self._new_glicko_sigma
        self.glicko_mu_hist.append((game_num, self.glicko_mu))
        self.glicko_phi_hist.append((game_num, self.glicko_phi))

    def _glicko_g(self, phi):
        """Reduce impact based on RD"""
        return 1 / np.sqrt(1 + (3 * phi * phi) / (np.pi * np.pi))

    def _glicko_E(self, mu, other_mu, other_phi):
        """Expected score"""
        return 1 / (1 + np.exp(-self._glicko_g(other_phi) * (mu - other_mu)))

    def _glicko_f(self, x, Delta, phi, v, a):
        """f(x) from sigma iteration step 1"""
        ex = np.power(np.e, x)
        numer1 = ex * (Delta * Delta - phi * phi - v - ex)
        denom1 = 2 * np.power(phi * phi + v * ex, 2)
        return (numer1 / denom1) - (x - a) / (self.glicko_tau*self.glicko_tau)

    def _glicko_sigma_iter(self, sigma, phi, Delta, v):
        # Step 2
        a = np.log(sigma * sigma)
        A = a
        if (Delta * Delta) > (phi * phi + v):
            B = np.log(Delta*Delta - phi*phi - v)
        else:
            k = 1
            while self._glicko_f((a-k*self.glicko_tau), Delta, phi, v, a) < 0:
                k += 1
            B = a - k * self.glicko_tau
        # Step 3
        fA = self._glicko_f(A, Delta, phi, v, a)
        fB = self._glicko_f(B, Delta, phi, v, a)
        # Step 4
        while abs(B - A) > self.glicko_epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = self._glicko_f(C, Delta, phi, v, a)
            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA = fA / 2
            B = C
            fB = fC
        return np.power(np.e, A / 2)


class Game:
    def __init__(self, name):
        self.name = name
        self.times_played = 0
        self.player_counts = {}

    def __lt__(self, other):
        if isinstance(other, Game):
            if self.times_played == other.times_played:
                return self.name < other.name
            return self.times_played < other.times_played
        raise NotImplemented()


class Play:
    def __init__(self, game, date, players):
        self.game = game
        self.date = date
        self.ranking = tuple(PlayerScore(p[0], p[1])
            for p in sorted(players.items(), key=lambda p: p[1], reverse=True))


def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = 'results.yml'
    print(f'Processing {filename}')
    with open(filename) as f:
        data = yaml.load(f)

    players = {}
    dates = []
    games = {}
    plays = []
    positions = {}
    head2head = {}
    total_games = len(data)

    for game in reversed(data):
        print(f"Processing game from {game['date']}")
        if 'disable' in game.keys() and game['disable']:
            print("Game is disabled")
            continue
        dates.append(game['date'])
        # Register a play of this game
        game_name = str(game['game'])
        if game_name not in games:
            games[game_name] = Game(game_name)
        games[game_name].times_played += 1
        player_count = len(game['players'])
        if player_count not in games[game_name].player_counts:
            games[game_name].player_counts[player_count] = 0
        games[game_name].player_counts[player_count] += 1

        # Record the information of the play
        play = Play(games[game_name], game['date'], game['players'])
        plays.append(play)
        game_num = len(plays)

        # Record information for all players
        for name, score in game['players'].items():
            name = name.lower()
            if name not in players:
                players[name] = Player(name, dates[0])
            players[name].scores.append(score)
            players[name].played += 1

        print('  Rank of players:', ', '.join((p.name.title()
                                               for p in play.ranking)))

        # Increase number of wins for the winning player
        players[play.ranking[0].name].wins += 1

        # Amounts of times finished in each game type
        if player_count not in positions:
            positions[player_count] = {}
        for i in range(len(play.ranking)):
            player_name = play.ranking[i].name
            if player_name not in positions[player_count]:
                positions[player_count][player_name] = {}
            try:
                positions[player_count][player_name][i+1] += 1
            except KeyError:
                positions[player_count][player_name][i+1] = 1

        # Calculate head to head scores
        for i in play.ranking:
            for j in play.ranking[play.ranking.index(i)+1:]:
                if i.name < j.name:
                    player1, player2 = i, j
                else:
                    player1, player2 = j, i
                pair = (player1.name, player2.name)
                if pair not in head2head:
                    head2head[pair] = [0, 0]
                head2head[pair][0] += _determine_score((player1,player2), 0, 1)
                head2head[pair][1] += _determine_score((player1,player2), 1, 0)

        # Calculate ELO
        for player in range(len(play.ranking)):
            scores = []
            elos = []
            glicko_mus = []
            glicko_phis = []
            for other in range(len(play.ranking)):
                if other == player:
                    continue
                elos.append(players[play.ranking[other].name].elo)

                # Determine win/loss against each opponent
                scores.append(_determine_score(play.ranking, player, other))

            # Calculate new score (but don't update)
            players[play.ranking[player].name].calculate_new_elo(elos, scores)
        # Update ELO
        for player in play.ranking:
            players[player.name].update_elo(play.date)
            print("  {} new ELO: {:.0f}".format(player.name.title(),
                                                players[player.name].elo))

    # Update glicko
    end_date = periodic_glicko(plays, players)

    # Give each player a unique color
    subdivs = math.ceil(math.pow(len(players), 1/3))
    subcubes = int(math.pow(subdivs, 3))
    cube_stride = subdivs - 1
    assert math.gcd(cube_stride, subcubes) == 1, \
           "stride is not coprime with number of color cubes"
    color_stride = math.floor(255 / (subdivs - 1))
    for i, (name, player) in enumerate(players.items()):
        cube = (i * cube_stride) % subcubes
        r = (cube % subdivs) * color_stride
        g = math.floor((cube / subdivs) % subdivs) * color_stride
        b = math.floor((cube / subdivs / subdivs) % subdivs) * color_stride
        player.color = f'#{r:0>2x}{g:0>2x}{b:0>2x}'

    # Determine begin and end dates for display
    plays.sort(key=lambda p: p.date)
    begin_date = date(plays[0].date.year, plays[0].date.month, 1)

    # Export results
    OUTDIR.mkdir()
    html_results(OUTDIR.child('index.html'), players, games, total_games,
                 dates, plays, positions, head2head)
    plot_elo(players, begin_date, end_date)
    plot_glicko(players, begin_date, end_date)
    plot_glicko_gaussians(players)

def periodic_glicko(plays, players):
    period = timedelta(days=28)
    plays.sort(key=lambda p: p.date)
    period_start = plays[0].date
    period_players = {}

    while period_start <= plays[-1].date:
        print('Calculating Glicko period', period_start, '-',
            period_start + period)
        f = lambda p: period_start <= p.date < period_start + period
        period_plays = list(filter(f, plays))
        # Register results of all games in this period
        for play in period_plays:
            for player in range(len(play.ranking)):
                player_name = play.ranking[player].name
                if player_name not in period_players:
                    period_players[player_name] = {
                        'name': play.ranking[player],
                        'mus': [],
                        'phis': [],
                        'scores': [],
                    }
                for other in range(len(play.ranking)):
                    if other == player:
                        continue
                    period_players[player_name]['mus'].append(
                        players[play.ranking[other].name].glicko_mu)
                    period_players[player_name]['phis'].append(
                        players[play.ranking[other].name].glicko_phi)
                    # Determine win/loss against opponent
                    period_players[player_name]['scores'].append(
                        _determine_score(play.ranking, player, other))
        # Calculate the scores in this period
        print('Games this period:', len(period_plays))
        if period_players.values():
            print('Duels this period:',
                max(len(p['mus']) for p in period_players.values()))
        else:
            print('No duels this period')
        _periodic_glicko_update(players, period_players, period_start + period)
        period_start += period
        period_players = {}
    return period_start  # End date of the last period

def _periodic_glicko_update(players, player_scores, game_num):
    for player, results in player_scores.items():
        players[player].calculate_new_glicko(results['mus'], results['phis'],
                                             results['scores'])
    for player in players:
        if player not in player_scores:
            print(f'    {player.title()} was inactive in period')
            players[player].calculate_glicko_inactive()
        players[player].update_glicko(game_num)
        print("  {} new Glicko: {:.2f}±{:.2f}".format(player.title(),
              players[player].glicko_mu * GLICKO_FACTOR + 1500,
              players[player].glicko_phi * GLICKO_FACTOR))

def _determine_score(ranking, player, opponent):
    diff = ranking[player].score - ranking[opponent].score
    if diff < -DRAW_DELTA:
        return LOSS
    elif diff > DRAW_DELTA:
        return WIN
    return DRAW

def plot_elo(players, begin_date, end_date):
    print('Plotting ELO scores')
    plt.clf()
    labels = []
    fig, ax = plt.subplots(1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for player in players:
        elo = np.array(players[player].elo_history).T
        line, = plt.plot(*elo, label=player.title(),
                         color=players[player].color)
        labels.append(line)
    ax.set_xlim(begin_date, end_date)
    fig.autofmt_xdate()
    plt.title('History of ELO ratings')
    plt.xlabel('Games played')
    plt.ylabel('ELO rating')
    plt.legend(handles=labels, loc=2)
    plt.savefig(OUTDIR.child('rankings_elo.png'))

def plot_glicko(players, begin_date, end_date):
    print('Plotting Glicko history')
    plt.clf()
    labels = []
    fig, ax = plt.subplots(1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    for player in players:
        # plot
        glicko_mu = np.array(players[player].glicko_mu_hist).T
        glicko_mu[1] = glicko_mu[1] * GLICKO_FACTOR + 1500
        line, = plt.plot(*glicko_mu, label=player.title(),
                         color=players[player].color)
        labels.append(line)

    ax.set_xlim(begin_date, end_date)
    ax.set_ylim(1200, 1900)
    fig.autofmt_xdate()
    plt.title('History of Glicko ratings')
    plt.xlabel('Games played')
    plt.ylabel('Glicko rating')
    plt.legend(handles=labels, loc=2)
    plt.savefig(OUTDIR.child('rankings_glicko.png'))

def plot_glicko_gaussians(players):
    print('Plotting current Glicko scores')
    plt.clf()
    labels = []
    ax = plt.gca()
    for name, player in players.items():
        mu = player.glicko_mu * GLICKO_FACTOR + 1500
        sigma = player.glicko_phi * GLICKO_FACTOR
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        line, = plt.plot(x, mlab.normpdf(x, mu, sigma), label=name.title(),
                         color=player.color)
        labels.append(line)
    ax.set_xlim(1000, 2000)
    ax.get_yaxis().set_ticks([])
    plt.title('Current Glicko ratings')
    plt.xlabel('Rating')
    plt.legend(handles=labels, loc=2)
    plt.savefig(OUTDIR.child('rankings_glicko_gaussians.png'))

def html_results(filename, players, games, total_games, dates, plays,
                 positions, head2head):
    env = Environment(
        loader=FileSystemLoader('templates'),
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template('rankings.html.j2')
    dates.sort()
    with open(filename, 'w') as f:
        f.write(template.render(players=players, games=games,
                                dates=dates, plays=plays,
                                GLICKO_FACTOR=GLICKO_FACTOR,
                                positions=positions,
                                head2head=head2head))

    copyfile('templates/style.css', 'output/style.css')

if __name__ == '__main__':
    main()
