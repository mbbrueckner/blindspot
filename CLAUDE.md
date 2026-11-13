# Blind Spot: Arbeitsregeln für Claude

Hinderniswarner für Blinde: iPhone-Kamera auf Brusthöhe, monokulare Tiefenschätzung, Stereo-Ton als Warnung.
Gruppe: Mika Brückner, Yannik Köllmann. Planung: `planning/ap/BlindSpot_Arbeitspakete.tex` (Arbeitspakete, Termine) und `planning/BlindSpot_Gantt.xlsx`.
Arbeitspaket-Nummern (z. B. 1.2, 5.1) stehen in Commit-Nachrichten und Docstrings, damit man Code und Plan zuordnen kann. Im englischen Code heißt ein Arbeitspaket "WP" (work package), z. B. "WP 5.1".

## Entscheidungen

- **Modellvergleich (AP 2.1): zwei bis drei Tiefenmodelle werden getestet, MiDaS ist eines davon.** Das Basismodell wird erst nach dem Vergleich auf den Testdaten festgelegt. Code muss deshalb modellunabhängig sein (Variante = Funktion `rgb -> (Tiefe, Alerts)`). Zu beachten: MiDaS liefert nur *relative* Tiefe (Skala und Verschiebung unbekannt), metrische Modelle liefern Meter direkt. Für den fairen Vergleich braucht MiDaS Bodenskalierung (AP 2.2) oder eine Skalierung gegen die Referenz, das im Logbuch dokumentieren. Die weiteren Kandidaten stehen noch nicht fest.
- **LiDAR-Zugang prüfen, bevor M1 fällig ist:** Es ist noch nicht getestet, ob wir vom iPhone 12 Pro LiDAR-Tiefe samt Intrinsics exportieren können (z. B. ARKit-Eigenbau oder eine Aufnahme-App). Ohne das fallen AP 1.3 und 1.1 aus. Ergebnis im Logbuch von M1 festhalten.
- Der nächste Termin mit dem Modulleiter (Niklas) ist um eine Woche verschoben. Die Meilenstein-Daten in der Planung wurden dabei noch nicht angepasst.

## Code-Regeln

- **Alles im Code ist Englisch:** Bezeichner, Docstrings, Kommentare, Fehlermeldungen, Konsolenausgaben, Dateinamen und Ordner (z. B. `planning/calibration/camera.json`). Nur Dokumentation für Niklas und die Gruppe (`CLAUDE.md`, `docs/`) bleibt Deutsch.
- **Docstrings sind Pflicht** für jedes Modul, jede Klasse und jede öffentliche Funktion: ein Satz, was sie tut (Imperativ-Stil wie "Reads ...", "Computes ..."), dazu Args/Returns mit Einheiten (meters, pixels). Das Repo ist Teil der Abgabe.
- Python in `.venv` (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`). Tests mit `.venv/bin/pytest`.
- **Codestil:** PEP 8, geprüft und formatiert mit [Ruff](https://docs.astral.sh/ruff/) (Konfiguration in `pyproject.toml`: Zeilenlänge 110, Importreihenfolge, Pflicht-Docstrings im Google-Stil, Tests ausgenommen). Vor jedem Commit `.venv/bin/ruff format . && .venv/bin/ruff check .` und `.venv/bin/pytest`. Typhinweise an öffentlichen Funktionen. `snake_case` für Funktionen und Variablen, `CamelCase` für Klassen.
- Tiefen immer in Metern als `float32`, ungültige Pixel sind `NaN`. Bilder sind RGB `uint8`, Form `(H, W, 3)`.
- Keine Daten im Repo: `data/` und `results/` stehen in `.gitignore`. Nur Splits, Annotationen und Kalibrierung (kleine Textdateien) werden versioniert.
- Neue Logik mit Verzweigungen bekommt einen kleinen Test in `tests/`.

## Git-Workflow (Vorschlag, noch nicht mit Mika abgestimmt)

GitHub Flow, passend für zwei Personen und drei Monate:

- `main` ist immer lauffähig (Tests grün). Direkte Pushes auf `main` vermeiden.
- Pro Arbeitspaket ein Branch: `ap-<Nr>-<kurzname>`, z. B. `ap-5.1-evaluation`. Kleine Änderungen sind erlaubt, aber die AP-Nummer gehört in den Branchnamen.
- Änderungen kommen per **Pull Request**, den die andere Person prüft (ein Approve reicht), dann **Squash-Merge** und Branch löschen. Die Person, die den Code geschrieben hat, merged nicht selbst.
- Commit-Nachrichten: englische Kurzzeile im Imperativ mit WP-Nummer, z. B. `WP 5.1: Add evaluation framework`. Ein Commit macht eine Sache.
- Meilensteine sind Tags auf `main` (`m1`, `m2`, …), siehe unten.
- Keine Daten, Modellgewichte oder Zugangsdaten im Repo. Große Dateien liegen außerhalb (Pfad im README).
- Branch-Schutz für `main` (PR nötig) stellt ihr in den GitHub-Einstellungen selbst ein, das kann ich nicht.

## Dokumentation pro Meilenstein (wichtig)

Niklas (Modulleiter) verlangt zu jedem Meilenstein **1 bis max. 2 Seiten Text**: Stand des Projekts und neu gewonnene Erkenntnisse. Das ist die Grundlage für Vortrag und Abschlussbericht, der Text darf dort wiederverwendet werden.
Deshalb sammeln wir laufend, nicht erst vor dem Termin.

- Pro Meilenstein gibt es `docs/meilensteine/M<n>.md` (Vorlage: `docs/meilensteine/VORLAGE.md`). Die Datei hat oben den **Abgabetext** (höchstens 2 Seiten, etwa 600 Wörter) und unten das **Logbuch** als Rohmaterial.
- **Claude pflegt das Logbuch mit.** Nach jeder abgeschlossenen Aufgabe (und beim Ende einer Sitzung) kommt ein Eintrag dazu: Datum, AP-Nummer, was getan wurde, was geklappt hat, was nicht geklappt hat und warum, Zahlen und Messwerte, offene Fragen. Fehlschläge gehören ausdrücklich hinein, sie sind oft die besten Erkenntnisse.
- Beim Meilenstein verdichtet Claude das Logbuch zum Abgabetext. Neue Erkenntnisse stehen in einem eigenen Abschnitt, damit sie sich später in den Abschlussbericht übernehmen lassen.
- **Git-Tag pro Meilenstein** (`m1`, `m2`, …) auf den Stand, der abgegeben wird. Claude setzt Tags und pusht sie nur auf Zuruf.
