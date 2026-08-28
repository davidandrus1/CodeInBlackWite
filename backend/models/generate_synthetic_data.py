"""
generate_synthetic_data.py

Generează date sintetice JSON în formatul exact Wyscout:
  1. synthetic_players.json  → format identic cu players (1).json
  2. Liga2_Match_*_players_stats.json → format identic cu match stats

Jucătorii sunt din Liga 2 România fictivă.
Rulează din backend/:
    python models/generate_synthetic_data.py
"""

import json
import os
import random
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_TEAMS        = 10
PLAYERS_PER_TEAM = 22
N_MATCHES_PER_TEAM = 30   # sezon complet
LIGA2_SCALE    = 0.80     # jucătorii din Liga 2 sunt ~80% față de Liga 1
START_PLAYER_ID = 9_000_001
START_TEAM_ID   = 900_001
START_MATCH_ID  = 8_000_001

OUTPUT_DIR = r"C:\Python\CodeInBlackWite\Date - meciuri"

# ─────────────────────────────────────────────
# ECHIPE LIGA 2 FICTIVE
# ─────────────────────────────────────────────
LIGA2_TEAMS = [
    {"id": START_TEAM_ID + i, "name": name}
    for i, name in enumerate([
        "FC Brașov", "CSM Iași", "Rapid II", "Steaua II",
        "Politehnica Timișoara", "Gloria Buzău", "FC Argeș II",
        "CSO Filiași", "Unirea Slobozia", "FC Hunedoara",
    ])
]

# ─────────────────────────────────────────────
# NUME ROMÂNEȘTI
# ─────────────────────────────────────────────
FIRST_NAMES = [
    "Alexandru", "Andrei", "Mihai", "Cristian", "Daniel", "David",
    "Florin", "Gabriel", "George", "Ioan", "Ionut", "Lucian",
    "Marius", "Mihnea", "Nicolae", "Paul", "Radu", "Razvan",
    "Robert", "Sebastian", "Stefan", "Tudor", "Vlad", "Adrian",
    "Bogdan", "Catalin", "Cosmin", "Denis", "Eduard", "Felix",
]
LAST_NAMES = [
    "Pop", "Ionescu", "Constantin", "Gheorghe", "Popa", "Lazar",
    "Dumitru", "Stan", "Stoica", "Moldovan", "Dinu", "Marin",
    "Nistor", "Matei", "Tanase", "Oprea", "Coman", "Iancu",
    "Rusu", "Ciobanu", "Mihai", "Vlad", "Barbu", "Enache",
    "Florea", "Popescu", "Dobre", "Tudor", "Neagu", "Costea",
]

# ─────────────────────────────────────────────
# POZITII
# ─────────────────────────────────────────────
POSITION_DISTRIBUTION = [
    ("gk",  "Goalkeeper", "GK", "GKP", 2),
    ("cb",  "Defender",   "DF", "DEF", 5),
    ("lb",  "Defender",   "DF", "DEF", 2),
    ("rb",  "Defender",   "DF", "DEF", 2),
    ("dmf", "Midfielder", "MD", "MID", 3),
    ("cmf", "Midfielder", "MD", "MID", 3),
    ("amf", "Midfielder", "MD", "MID", 2),
    ("cf",  "Forward",    "FW", "FWD", 2),
    ("lw",  "Forward",    "FW", "FWD", 1),
]

def pick_position():
    choices = [(code, name, c2, c3) for code, name, c2, c3, w in POSITION_DISTRIBUTION
               for _ in range(w)]
    return random.choice(choices)

# ─────────────────────────────────────────────
# DISTRIBUȚII STATS PER POZIȚIE (Liga 2 = Liga 1 × 0.8)
# ─────────────────────────────────────────────
POS_STATS = {
    "gk":  {"minutes": (70, 90), "passes": (20, 45), "saves": (1, 6),
             "exits": (0, 3), "aerialDuels": (0, 4)},
    "cb":  {"minutes": (70, 90), "passes": (30, 55), "duels": (4, 12),
             "interceptions": (1, 6), "clearances": (2, 8), "aerialDuels": (2, 7)},
    "lb":  {"minutes": (65, 90), "passes": (25, 50), "crosses": (0, 4),
             "duels": (3, 10), "progressiveRun": (0, 3)},
    "rb":  {"minutes": (65, 90), "passes": (25, 50), "crosses": (0, 4),
             "duels": (3, 10), "progressiveRun": (0, 3)},
    "dmf": {"minutes": (65, 90), "passes": (35, 65), "duels": (4, 12),
             "interceptions": (1, 5), "ballRecoveries": (2, 8)},
    "cmf": {"minutes": (60, 90), "passes": (30, 60), "keyPasses": (0, 3),
             "duels": (3, 10), "dribbles": (0, 4)},
    "amf": {"minutes": (55, 90), "passes": (25, 50), "keyPasses": (0, 4),
             "shots": (0, 3), "dribbles": (0, 5), "xgAssist": (0, 0.3)},
    "cf":  {"minutes": (55, 90), "passes": (15, 35), "shots": (1, 5),
             "duels": (3, 10), "touchInBox": (1, 6), "xgShot": (0.05, 0.4)},
    "lw":  {"minutes": (50, 85), "passes": (20, 40), "shots": (0, 4),
             "dribbles": (1, 6), "crosses": (0, 4), "progressiveRun": (0, 4)},
}

def r(lo, hi, scale=1.0):
    """Random float în interval, scalat pentru Liga 2."""
    lo2 = lo * scale
    hi2 = hi * scale
    return round(max(0, random.gauss((lo2+hi2)/2, (hi2-lo2)/4)), 2)

def ri(lo, hi, scale=1.0):
    return int(max(0, round(r(lo, hi, scale))))

# ─────────────────────────────────────────────
# GENERARE STATS MECI
# ─────────────────────────────────────────────
def generate_match_stats(pos_code: str, minutes: int) -> dict:
    s  = POS_STATS.get(pos_code, POS_STATS["cmf"])
    sc = LIGA2_SCALE

    passes            = ri(s.get("minutes", (60,90))[0]//3, s.get("passes", (20,40))[1], sc)
    successfulPasses  = ri(passes * 0.70, passes * 0.92)
    keyPasses         = ri(0, s.get("keyPasses", (0,2))[1], sc)
    successfulKeyPasses = ri(0, keyPasses)
    shots             = ri(0, s.get("shots", (0,2))[1], sc)
    shotsOnTarget     = ri(0, shots)
    goals             = ri(0, shotsOnTarget // 2)
    assists           = ri(0, min(1, keyPasses))
    xgShot            = round(random.uniform(0, s.get("xgShot", (0, 0.2))[1] * sc), 3)
    xgAssist          = round(random.uniform(0, s.get("xgAssist", (0, 0.15))[1] * sc), 3)
    duels             = ri(s.get("duels", (2,8))[0], s.get("duels", (2,8))[1], sc)
    duelsWon          = ri(duels * 0.3, duels * 0.6)
    defensiveDuels    = ri(duels // 2, duels)
    defensiveDuelsWon = ri(0, defensiveDuels)
    offensiveDuels    = ri(0, duels - defensiveDuels)
    offensiveDuelsWon = ri(0, offensiveDuels)
    aerialDuels       = ri(0, s.get("aerialDuels", (0,3))[1], sc)
    aerialDuelsWon    = ri(0, aerialDuels)
    interceptions     = ri(0, s.get("interceptions", (0,3))[1], sc)
    clearances        = ri(0, s.get("clearances", (0,4))[1], sc)
    dribbles          = ri(0, s.get("dribbles", (0,3))[1], sc)
    successfulDribbles = ri(0, dribbles)
    crosses           = ri(0, s.get("crosses", (0,3))[1], sc)
    successfulCrosses = ri(0, crosses)
    progressiveRun    = ri(0, s.get("progressiveRun", (0,2))[1], sc)
    progressivePasses = ri(0, passes // 5)
    successfulProgressivePasses = ri(0, progressivePasses)
    forwardPasses     = ri(passes // 4, passes // 2)
    successfulForwardPasses = ri(0, forwardPasses)
    backPasses        = ri(passes // 6, passes // 3)
    successfulBackPasses = ri(backPasses * 0.7, backPasses)
    lateralPasses     = max(0, passes - forwardPasses - backPasses)
    successfulLateralPasses = ri(lateralPasses * 0.7, lateralPasses)
    passesToFinalThird = ri(0, passes // 6)
    successfulPassesToFinalThird = ri(0, passesToFinalThird)
    ballRecoveries    = ri(0, s.get("ballRecoveries", (1,5))[1], sc)
    counterpressingRecoveries = ri(0, ballRecoveries // 2)
    opponentHalfRecoveries = ri(0, ballRecoveries // 2)
    losses            = ri(1, 8, sc)
    ownHalfLosses     = ri(0, losses // 2)
    dangerousOwnHalfLosses = ri(0, ownHalfLosses // 2)
    touchInBox        = ri(0, s.get("touchInBox", (0,3))[1], sc)
    smartPasses       = ri(0, keyPasses)
    successfulSmartPasses = ri(0, smartPasses)
    throughPasses     = ri(0, 2)
    successfulThroughPasses = ri(0, throughPasses)
    verticalPasses    = ri(0, passes // 4)
    successfulVerticalPasses = ri(0, verticalPasses)
    longPasses        = ri(0, passes // 5)
    successfulLongPasses = ri(0, longPasses)
    shotAssists       = ri(0, keyPasses)
    linkupPlays       = ri(0, 4, sc)
    successfulLinkupPlays = ri(0, linkupPlays)
    accelerations     = ri(0, 3, sc)
    fouls             = ri(0, 3, sc)
    looseBallDuels    = ri(0, 3, sc)
    looseBallDuelsWon = ri(0, looseBallDuels)
    slidingTackles    = ri(0, 2, sc)
    successfulSlidingTackles = ri(0, slidingTackles)
    dribblesAgainst   = ri(0, 3, sc)
    dribblesAgainstWon = ri(0, dribblesAgainst)
    pressingDuels     = ri(0, 3, sc)
    pressingDuelsWon  = ri(0, pressingDuels)
    newDuelsWon       = ri(duelsWon, duelsWon + 2)
    newDefensiveDuelsWon = ri(0, newDuelsWon)
    newOffensiveDuelsWon = ri(0, max(0, newDuelsWon - newDefensiveDuelsWon))
    fieldAerialDuels  = aerialDuels
    fieldAerialDuelsWon = aerialDuelsWon

    # GK specific
    is_gk = pos_code == "gk"
    gkSaves          = ri(1, 5) if is_gk else 0
    gkConcededGoals  = ri(0, 2) if is_gk else 0
    gkShotsAgainst   = gkSaves + gkConcededGoals if is_gk else 0
    gkExits          = ri(0, 3) if is_gk else 0
    gkSuccessfulExits = ri(0, gkExits) if is_gk else 0
    gkAerialDuels    = ri(0, 4) if is_gk else 0
    gkAerialDuelsWon = ri(0, gkAerialDuels) if is_gk else 0
    goalKicks        = ri(2, 8) if is_gk else 0
    goalKicksShort   = ri(0, goalKicks // 2) if is_gk else 0
    goalKicksLong    = ri(0, goalKicks - goalKicksShort) if is_gk else 0
    successfulGoalKicks = ri(goalKicks * 0.5, goalKicks) if is_gk else 0
    xgSave           = round(random.uniform(0.5, 2.5), 3) if is_gk else 0

    total = {
        "matches": 1, "matchesInStart": 1, "matchesSubstituted": 0,
        "matchesComingOff": 0, "minutesOnField": minutes, "minutesTagged": minutes,
        "goals": goals, "assists": assists, "shots": shots,
        "headShots": ri(0, shots // 2), "yellowCards": ri(0, 1),
        "redCards": 0, "directRedCards": 0, "penalties": 0,
        "linkupPlays": linkupPlays, "duels": duels, "duelsWon": duelsWon,
        "defensiveDuels": defensiveDuels, "defensiveDuelsWon": defensiveDuelsWon,
        "offensiveDuels": offensiveDuels, "offensiveDuelsWon": offensiveDuelsWon,
        "aerialDuels": aerialDuels, "aerialDuelsWon": aerialDuelsWon,
        "fouls": fouls, "passes": passes, "successfulPasses": successfulPasses,
        "smartPasses": smartPasses, "successfulSmartPasses": successfulSmartPasses,
        "passesToFinalThird": passesToFinalThird,
        "successfulPassesToFinalThird": successfulPassesToFinalThird,
        "crosses": crosses, "successfulCrosses": successfulCrosses,
        "forwardPasses": forwardPasses, "successfulForwardPasses": successfulForwardPasses,
        "backPasses": backPasses, "successfulBackPasses": int(successfulBackPasses),
        "throughPasses": throughPasses, "successfulThroughPasses": successfulThroughPasses,
        "keyPasses": keyPasses, "successfulKeyPasses": successfulKeyPasses,
        "verticalPasses": verticalPasses, "successfulVerticalPasses": successfulVerticalPasses,
        "longPasses": longPasses, "successfulLongPasses": successfulLongPasses,
        "dribbles": dribbles, "successfulDribbles": successfulDribbles,
        "interceptions": interceptions, "defensiveActions": defensiveDuels + interceptions,
        "successfulDefensiveAction": defensiveDuelsWon + interceptions,
        "attackingActions": shots + dribbles, "successfulAttackingActions": successfulDribbles + goals,
        "freeKicks": 0, "freeKicksOnTarget": 0, "directFreeKicks": 0,
        "directFreeKicksOnTarget": 0, "corners": 0, "successfulPenalties": 0,
        "successfulLinkupPlays": successfulLinkupPlays, "accelerations": accelerations,
        "pressingDuels": pressingDuels, "pressingDuelsWon": pressingDuelsWon,
        "looseBallDuels": looseBallDuels, "looseBallDuelsWon": looseBallDuelsWon,
        "missedBalls": 0, "shotAssists": shotAssists, "shotOnTargetAssists": ri(0, shotAssists),
        "recoveries": ballRecoveries + ri(0, 3), "opponentHalfRecoveries": opponentHalfRecoveries,
        "dangerousOpponentHalfRecoveries": ri(0, opponentHalfRecoveries // 2),
        "losses": losses, "ownHalfLosses": ownHalfLosses,
        "dangerousOwnHalfLosses": dangerousOwnHalfLosses,
        "xgShot": xgShot, "xgAssist": xgAssist, "xgSave": xgSave,
        "receivedPass": ri(15, 40), "touchInBox": touchInBox,
        "progressiveRun": progressiveRun, "offsides": ri(0, 2) if pos_code in ["cf", "lw"] else 0,
        "clearances": clearances, "secondAssists": 0, "thirdAssists": 0,
        "shotsBlocked": ri(0, 2) if pos_code in ["cb", "dmf"] else 0,
        "foulsSuffered": ri(0, 3), "progressivePasses": progressivePasses,
        "counterpressingRecoveries": counterpressingRecoveries,
        "slidingTackles": slidingTackles, "goalKicks": goalKicks,
        "dribblesAgainst": dribblesAgainst, "dribblesAgainstWon": dribblesAgainstWon,
        "goalKicksShort": goalKicksShort, "goalKicksLong": goalKicksLong,
        "shotsOnTarget": shotsOnTarget, "successfulProgressivePasses": successfulProgressivePasses,
        "successfulSlidingTackles": successfulSlidingTackles,
        "successfulGoalKicks": successfulGoalKicks,
        "fieldAerialDuels": fieldAerialDuels, "fieldAerialDuelsWon": fieldAerialDuelsWon,
        "gkCleanSheets": 0, "gkConcededGoals": gkConcededGoals,
        "gkShotsAgainst": gkShotsAgainst, "gkExits": gkExits,
        "gkSuccessfulExits": gkSuccessfulExits, "gkAerialDuels": gkAerialDuels,
        "gkAerialDuelsWon": gkAerialDuelsWon, "gkSaves": gkSaves,
        "newDuelsWon": newDuelsWon, "newDefensiveDuelsWon": newDefensiveDuelsWon,
        "newOffensiveDuelsWon": newOffensiveDuelsWon, "newSuccessfulDribbles": successfulDribbles,
        "lateralPasses": lateralPasses, "successfulLateralPasses": int(successfulLateralPasses),
    }

    # Average (per 90)
    f = 90 / max(minutes, 1)
    average = {k: round(v * f, 2) for k, v in total.items()
               if k not in ["matches","matchesInStart","matchesSubstituted",
                             "matchesComingOff","minutesOnField","minutesTagged",
                             "yellowCards","redCards","directRedCards","penalties"]}
    average["passLength"]     = round(random.uniform(15, 30), 2)
    average["longPassLength"] = round(random.uniform(25, 45), 2)
    average["ballLosses"]     = losses
    average["ballRecoveries"] = round(ballRecoveries * f, 2)

    # Percent
    def pct(num, den):
        return round(num / den * 100, 2) if den > 0 else 0.0

    percent = {
        "duelsWon":                    pct(duelsWon, duels),
        "defensiveDuelsWon":           pct(defensiveDuelsWon, defensiveDuels),
        "offensiveDuelsWon":           pct(offensiveDuelsWon, offensiveDuels),
        "aerialDuelsWon":              pct(aerialDuelsWon, aerialDuels),
        "successfulPasses":            pct(successfulPasses, passes),
        "successfulSmartPasses":       pct(successfulSmartPasses, smartPasses),
        "successfulPassesToFinalThird": pct(successfulPassesToFinalThird, passesToFinalThird),
        "successfulCrosses":           pct(successfulCrosses, crosses),
        "successfulDribbles":          pct(successfulDribbles, dribbles),
        "shotsOnTarget":               pct(shotsOnTarget, shots),
        "headShotsOnTarget":           0.0,
        "goalConversion":              pct(goals, shots),
        "directFreeKicksOnTarget":     0.0, "penaltiesConversion": 0.0, "win": 0.0,
        "successfulForwardPasses":     pct(successfulForwardPasses, forwardPasses),
        "successfulBackPasses":        pct(int(successfulBackPasses), backPasses),
        "successfulThroughPasses":     pct(successfulThroughPasses, throughPasses),
        "successfulKeyPasses":         pct(successfulKeyPasses, keyPasses),
        "successfulVerticalPasses":    pct(successfulVerticalPasses, verticalPasses),
        "successfulLongPasses":        pct(successfulLongPasses, longPasses),
        "successfulShotAssists":       0.0,
        "successfulLinkupPlays":       pct(successfulLinkupPlays, linkupPlays),
        "yellowCardsPerFoul":          0.0,
        "successfulProgressivePasses": pct(successfulProgressivePasses, progressivePasses),
        "successfulSlidingTackles":    pct(successfulSlidingTackles, slidingTackles),
        "successfulGoalKicks":         pct(successfulGoalKicks, goalKicks),
        "dribblesAgainstWon":          pct(dribblesAgainstWon, dribblesAgainst),
        "fieldAerialDuelsWon":         pct(fieldAerialDuelsWon, fieldAerialDuels),
        "gkSaves":                     pct(gkSaves, gkShotsAgainst),
        "gkSuccessfulExits":           pct(gkSuccessfulExits, gkExits),
        "gkAerialDuelsWon":            pct(gkAerialDuelsWon, gkAerialDuels),
        "newDuelsWon":                 pct(newDuelsWon, duels),
        "newDefensiveDuelsWon":        pct(newDefensiveDuelsWon, defensiveDuels),
        "newOffensiveDuelsWon":        pct(newOffensiveDuelsWon, offensiveDuels),
        "newSuccessfulDribbles":       pct(successfulDribbles, dribbles),
        "successfulLateralPasses":     pct(int(successfulLateralPasses), lateralPasses),
    }

    return {"total": total, "average": average, "percent": percent}


# ─────────────────────────────────────────────
# GENERARE JUCĂTORI
# ─────────────────────────────────────────────
def generate_players():
    players = []
    player_id = START_PLAYER_ID

    for team in LIGA2_TEAMS:
        # Distribuție poziții per echipă
        positions_for_team = []
        for code, name, c2, c3, count in POSITION_DISTRIBUTION:
            for _ in range(count):
                positions_for_team.append((code, name, c2, c3))

        # Completăm până la PLAYERS_PER_TEAM
        while len(positions_for_team) < PLAYERS_PER_TEAM:
            positions_for_team.append(("cmf", "Midfielder", "MD", "MID"))

        random.shuffle(positions_for_team)

        for pos_code, pos_name, c2, c3 in positions_for_team[:PLAYERS_PER_TEAM]:
            first = random.choice(FIRST_NAMES)
            last  = random.choice(LAST_NAMES)
            # Vârstă: GK mai mari, tineri la FW
            if pos_code == "gk":
                age = random.randint(20, 35)
            elif pos_code in ["cf", "lw"]:
                age = random.randint(18, 28)
            else:
                age = random.randint(19, 33)

            birth_year = 2025 - age
            birth_month = random.randint(1, 12)
            birth_day = random.randint(1, 28)
            birth_date = f"{birth_year}-{birth_month:02d}-{birth_day:02d}"

            height = random.randint(168, 196)
            weight = random.randint(60, 92)

            players.append({
                "wyId": player_id,
                "shortName": f"{first[0]}. {last}",
                "firstName": first,
                "middleName": "",
                "lastName": last,
                "height": height,
                "weight": weight,
                "birthDate": birth_date,
                "birthArea": {"id": 642, "alpha2code": "RO",
                              "alpha3code": "ROU", "name": "Romania"},
                "passportArea": {"id": 642, "alpha2code": "RO",
                                 "alpha3code": "ROU", "name": "Romania"},
                "role": {"name": pos_name, "code2": c2, "code3": c3},
                "foot": random.choice(["right", "left", None]),
                "currentTeamId": team["id"],
                "currentNationalTeamId": None,
                "gender": "male",
                "status": "active",
                "_pos_code": pos_code,  # folosit intern, ignorat de data_processor
            })
            player_id += 1

    return players


# ─────────────────────────────────────────────
# GENERARE MECIURI
# ─────────────────────────────────────────────
def generate_matches(players):
    teams = LIGA2_TEAMS
    match_id = START_MATCH_ID
    match_files = []

    # Programăm meciuri: fiecare echipă cu fiecare, tur-retur
    schedule = []
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            schedule.append((teams[i], teams[j]))
            schedule.append((teams[j], teams[i]))

    random.shuffle(schedule)

    # Grupăm jucătorii pe echipă
    team_players = {}
    for team in teams:
        team_players[team["id"]] = [p for p in players if p["currentTeamId"] == team["id"]]

    for home_team, away_team in schedule:
        home_players = team_players.get(home_team["id"], [])
        away_players = team_players.get(away_team["id"], [])

        if not home_players or not away_players:
            continue

        # Selectăm 11 titulari + 3 rezerve per echipă
        home_squad = random.sample(home_players, min(14, len(home_players)))
        away_squad = random.sample(away_players, min(14, len(away_players)))

        match_players = []
        comp_id  = 720  # Liga 2 fictivă
        season_id = 200000
        round_id  = 5000000 + match_id

        for i, player in enumerate(home_squad + away_squad):
            pos_code = player.get("_pos_code", "cmf")
            minutes  = random.randint(45, 90) if i >= 11 else random.randint(70, 90)
            if minutes < 45:
                continue

            stats = generate_match_stats(pos_code, minutes)

            match_players.append({
                "playerId":      player["wyId"],
                "matchId":       match_id,
                "competitionId": comp_id,
                "seasonId":      season_id,
                "roundId":       round_id,
                "positions": [{"position": {"name": player["role"]["name"],
                                            "code": pos_code},
                               "percent": 100}],
                "total":   stats["total"],
                "average": stats["average"],
                "percent": stats["percent"],
            })

        match_files.append({
            "match_id": match_id,
            "filename": f"Liga2_Match_{match_id}_players_stats.json",
            "data": {"players": match_players},
        })
        match_id += 1

    return match_files


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. Generând jucători sintetici...")
    players = generate_players()

    # Salvează players JSON (fără _pos_code intern)
    players_clean = [{k: v for k, v in p.items() if k != "_pos_code"} for p in players]
    players_json = {
        "meta": {"total_items": len(players_clean)},
        "players": players_clean,
    }

    # Adaugă la players (1).json existent SAU salvează separat
    players_out = os.path.join(OUTPUT_DIR, "players_synthetic.json")
    with open(players_out, "w", encoding="utf-8") as f:
        json.dump(players_json, f, ensure_ascii=False, indent=2)
    print(f"   → {len(players_clean)} jucători salvați în players_synthetic.json")

    print("2. Generând meciuri sintetice...")
    match_files = generate_matches(players)

    for mf in match_files:
        path = os.path.join(OUTPUT_DIR, mf["filename"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mf["data"], f, ensure_ascii=False)

    print(f"   → {len(match_files)} fișiere de meciuri salvate în Date - meciuri/")

    total_players_in_matches = sum(len(mf["data"]["players"]) for mf in match_files)
    print(f"\n{'='*50}")
    print(f"Jucători generați:        {len(players_clean)}")
    print(f"Meciuri generate:         {len(match_files)}")
    print(f"Înregistrări meci total:  {total_players_in_matches}")
    print(f"{'='*50}")
    print("\nPasul următor:")
    print("  1. Actualizează data_processor.py să citească și players_synthetic.json")
    print("  2. Rulează train.py din nou")
    print("  3. Rulează streamlit run app.py")