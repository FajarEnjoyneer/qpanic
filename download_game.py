import requests
import zipfile
import time
import re
from datetime import datetime

USERNAME = "qpanic"
BASE_URL = "https://api.chess.com/pub/player"
ZIP_NAME = f"{USERNAME}_chesscom_games_clean.zip"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChessComDownloader/1.0)"
}

# Header final (URUTAN DIJAGA)
HEADER_ORDER = [
    "Event",
    "Site",
    "Date",
    "Round",
    "White",
    "Black",
    "Result",
    "TimeControl",
    "WhiteElo",
    "BlackElo",
    "Termination",
    "ECO",
    "EndTime",
    "Link",
]

CLOCK_REGEX = re.compile(r"\{\[%clk [^}]+\]\}")

def get_archives(username):
    url = f"{BASE_URL}/{username}/games/archives"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["archives"]

def get_games(archive_url):
    r = requests.get(archive_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["games"]

def clean_pgn(pgn_text):
    lines = pgn_text.splitlines()

    headers = {}
    moves = []
    in_moves = False

    for line in lines:
        if line.startswith("[") and not in_moves:
            key = line.split(" ", 1)[0][1:]
            headers[key] = line
        elif line.strip() == "":
            in_moves = True
        else:
            in_moves = True
            moves.append(line)

    # === FORCE HEADER NORMALIZATION ===
    headers["Round"] = '[Round "?"]'

    if "EndTime" in headers and "GMT" not in headers["EndTime"]:
        value = headers["EndTime"].split('"')[1]
        headers["EndTime"] = f'[EndTime "{value} GMT+0000"]'

    # === CLEAN MOVES ===
    moves_text = " ".join(moves)

    # Hapus clock annotation
    moves_text = CLOCK_REGEX.sub("", moves_text)

    # Hapus format "1... e5"
    moves_text = re.sub(r"\d+\.\.\.\s*", "", moves_text)

    # Rapikan spasi
    moves_text = re.sub(r"\s+", " ", moves_text).strip()

    # === BUILD FINAL PGN ===
    output = []
    for key in HEADER_ORDER:
        if key in headers:
            output.append(headers[key])

    output.append("")
    output.append(moves_text)

    return "\n".join(output) + "\n"

def main():
    archives = get_archives(USERNAME)
    print(f"Total arsip: {len(archives)}")

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        total = 0

        for archive in archives:
            print(f"Ambil: {archive}")
            games = get_games(archive)

            for game in games:
                raw_pgn = game.get("pgn")
                if not raw_pgn:
                    continue

                white = game["white"]["username"]
                black = game["black"]["username"]
                date = datetime.utcfromtimestamp(
                    game["end_time"]
                ).strftime("%Y.%m.%d")
                game_id = game["url"].split("/")[-1]

                filename = f"{white}_vs_{black}_{date}_{game_id}.pgn"

                clean = clean_pgn(raw_pgn)
                zipf.writestr(filename, clean)

                total += 1
                time.sleep(0.1)

    print("\nSELESAI")
    print(f"Total game : {total}")
    print(f"Output ZIP : {ZIP_NAME}")

if __name__ == "__main__":
    main()
