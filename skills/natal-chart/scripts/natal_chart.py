#!/usr/bin/env python3
"""Natal chart calculator — verified against pyswisseph 2.10.3.

Usage:
  python3 natal_chart.py --date 1983-06-30 --time 16:20 --tz -5 --lat 39.0473 --lon -95.6752

All times UTC-offset aware: pass local time + numeric UTC offset (hours, west negative).
Output: planets, angles (Placidus), aspects. Deterministic — same inputs, same output.
"""
import argparse
import swisseph as swe

SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

BODIES = [
    ("Sun", swe.SUN), ("Moon", swe.MOON), ("Mercury", swe.MERCURY),
    ("Venus", swe.VENUS), ("Mars", swe.MARS), ("Jupiter", swe.JUPITER),
    ("Saturn", swe.SATURN), ("Uranus", swe.URANUS), ("Neptune", swe.NEPTUNE),
    ("Pluto", swe.PLUTO), ("NorthNode", swe.MEAN_NODE),
]

ASPECTS = [(0, "conjunction", 10), (60, "sextile", 6), (90, "square", 8),
           (120, "trine", 8), (180, "opposition", 10)]


def fmt(lon):
    s = int(lon // 30)
    d = lon % 30
    return f"{int(d)}°{SIGNS[s]} {int((d % 1) * 60):02d}'"


def house_of(lon, cusps):
    for i in range(12):
        lo, hi = cusps[i], cusps[(i + 1) % 12]
        if lo < hi and lo <= lon < hi:
            return i + 1
        if lo > hi and (lon >= lo or lon < hi):
            return i + 1
    return "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD (local)")
    p.add_argument("--time", required=True, help="HH:MM (local)")
    p.add_argument("--tz", required=True, type=float, help="UTC offset hours, e.g. -5 for CDT")
    p.add_argument("--lat", required=True, type=float)
    p.add_argument("--lon", required=True, type=float, help="west negative")
    p.add_argument("--houses", default="P", help="house system: P=Placidus K=Koch W=Whole")
    a = p.parse_args()

    y, m, d = map(int, a.date.split("-"))
    hh, mm = map(int, a.time.split(":"))
    jd = swe.julday(y, m, d, hh + mm / 60.0 - a.tz)

    cusps, ascmc = swe.houses(jd, a.lat, a.lon, a.houses.encode())

    print(f"# Natal chart  jd={jd:.5f}  lat={a.lat} lon={a.lon}  houses={a.houses}")
    print(f"Ascendant  {fmt(ascmc[0])}")
    print(f"MC         {fmt(ascmc[1])}")
    print()
    pos = {}
    for name, b in BODIES:
        r = swe.calc_ut(jd, b, swe.FLG_SWIEPH | swe.FLG_SPEED)
        lon_, speed = r[0][0], r[0][3]  # index 3 = dlon/day. NOT index 4 (that's dlat).
        pos[name] = lon_
        retro = " R" if speed < 0 else ""
        print(f"{name:<10} {fmt(lon_):<18} house {house_of(lon_, cusps)}{retro}")
    print()
    names = list(pos)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = abs(pos[names[i]] - pos[names[j]]) % 360
            if diff > 180:
                diff = 360 - diff
            for ang, aname, orb in ASPECTS:
                if abs(diff - ang) <= orb:
                    print(f"{names[i]:<10} {aname:<12} {names[j]:<10} orb {abs(diff - ang):.1f}")


if __name__ == "__main__":
    main()
