"""
체육수업 팀 편성 프로그램 (전략형 스포츠: 농구 · 배구 · 티볼)
- AHP 결과(전문가 10명, 최종) 기반 가중치 적용
- MILP 기반 균형 팀 편성 (Rubin & Bai, 2015 방식)
- 학생 참여형 입력 (학생이 직접 응답 → 교사 확인·수정 → 편성)
- 주장 사전 배정 (리더십 요인은 편성 요인에서 제외)

실행:  streamlit run team_assign.py
학생 참여 링크:  <앱 주소>/?code=참여코드
"""
import io
import hashlib
import os
import re
import json
import random
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import pulp
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

# ─── PuLP 버전 호환 (구버전·신버전 모두 동작) ─────────────────
def value(v):
    f = getattr(pulp, "value", None)
    if f is not None:
        return f(v)
    return v.value() if hasattr(v, "value") else v


def _make_solver(time_limit):
    """PuLP 내장 CBC 솔버 생성 (PuLP 버전별 인자명 차이 대응). 못 찾으면 None → PuLP 기본 솔버(CBC)"""
    for name in ("PULP_CBC_CMD", "COIN_CMD"):
        cls = getattr(pulp, name, None)
        if cls is None:
            continue
        for kw in ({"msg": False, "timeLimit": time_limit}, {"msg": False, "time_limit": time_limit},
                   {"msg": False}, {}):
            try:
                solver = cls(**kw)
            except Exception:
                continue
            try:
                if hasattr(solver, "available") and not solver.available():
                    break
            except Exception:
                pass
            return solver
    return None


def _solve_outcome(prob):
    """
    솔버 결과 판정 → "optimal" | "timeout" | "infeasible"
    PuLP 2.x에서는 CBC가 시간 제한으로 멈춰도 실행 가능한 해가 있으면 status=1(Optimal)을 반환하므로,
    sol_status(1=최적 증명, 2=실행 가능하나 최적 미증명)까지 확인하여 '최적해'만 통과시킨다.
    """
    stt = prob.status
    names = getattr(pulp, "LpStatus", None)
    label = names.get(stt, stt) if isinstance(names, dict) else stt
    label = str(getattr(label, "name", label)).lower()
    status_ok = stt == 1 or (label.endswith("optimal") and "not" not in label)
    sol = getattr(prob, "sol_status", None)
    feasible_only = getattr(pulp, "LpSolutionIntegerFeasible", 2)
    if status_ok and sol == feasible_only:
        return "timeout"          # 제한시간 내 최적성 미확인
    if status_ok:
        return "optimal"
    if stt == 0 or "not solved" in label:
        return "timeout"          # 제한시간 내 해를 찾지 못함
    return "infeasible"           # 제약조건을 동시에 만족하는 해 없음


# ════════════════════════════════════════════════════════════════
# 1. AHP 결과 (출처: AHP 결과 정리 엑셀 — 전문가 10명, 최종)
#    가중치: 기하평균법(응답자 기하평균 통합행렬 → 행 기하평균 정규화)
#    CR: 통합행렬 최대고유치 기준 / 종합 가중치 = 대분류 × 하위요인
#    ※ 소수점 셋째 자리 반올림 표기, 순위는 반올림 이전 값 기준(엑셀 그대로)
# ════════════════════════════════════════════════════════════════
MAIN_ORDER = ["신체 능력", "운동기능수준", "운동 경험", "인지적 능력", "사회·정서적 특성"]

# 하위요인: (대분류, 하위요인, 하위요인 가중치, 하위요인 순위, 종합 가중치, 종합 순위)
AHP = {
    "농구": {
        "main": {"운동기능수준": (0.422, 1), "운동 경험": (0.281, 2), "신체 능력": (0.130, 3),
                 "인지적 능력": (0.098, 4), "사회·정서적 특성": (0.068, 5)},
        "main_cr": 0.039,
        "sub_cr": {"신체 능력": 0.018, "운동기능수준": 0.001, "운동 경험": 0.023,
                   "인지적 능력": 0.040, "사회·정서적 특성": 0.022},
        "sub": [
            ("신체 능력", "심폐지구력", 0.189, 3, 0.025, 10),
            ("신체 능력", "근력·근지구력", 0.079, 4, 0.010, 17),
            ("신체 능력", "순발력", 0.340, 2, 0.044, 9),
            ("신체 능력", "민첩성", 0.392, 1, 0.051, 8),
            ("운동기능수준", "운동 학습력", 0.264, 3, 0.112, 4),
            ("운동기능수준", "운동 감각(운동 센스)", 0.456, 1, 0.193, 1),
            ("운동기능수준", "전술 실행력", 0.280, 2, 0.118, 3),
            ("운동 경험", "학원 수강 경력", 0.462, 1, 0.130, 2),
            ("운동 경험", "학교스포츠클럽활동 경력", 0.316, 2, 0.089, 5),
            ("운동 경험", "체육특기자 여부(종목 무관)", 0.222, 3, 0.063, 6),
            ("인지적 능력", "규칙 이해도", 0.240, 2, 0.024, 11),
            ("인지적 능력", "전술 이해도", 0.229, 3, 0.022, 13),
            ("인지적 능력", "상황 판단력", 0.531, 1, 0.052, 7),
            ("사회·정서적 특성", "리더십", 0.098, 5, 0.007, 18),
            ("사회·정서적 특성", "수업참여도", 0.335, 1, 0.023, 12),
            ("사회·정서적 특성", "협동심", 0.217, 2, 0.015, 14),
            ("사회·정서적 특성", "자신감", 0.188, 3, 0.013, 15),
            ("사회·정서적 특성", "도전의식/용기", 0.161, 4, 0.011, 16),
        ],
    },
    "배구": {
        "main": {"운동기능수준": (0.436, 1), "운동 경험": (0.271, 2), "신체 능력": (0.143, 3),
                 "인지적 능력": (0.088, 4), "사회·정서적 특성": (0.062, 5)},
        "main_cr": 0.051,
        "sub_cr": {"신체 능력": 0.022, "운동기능수준": 0.034, "운동 경험": 0.002,
                   "인지적 능력": 0.002, "사회·정서적 특성": 0.012},
        "sub": [
            ("신체 능력", "심폐지구력", 0.053, 4, 0.008, 17),
            ("신체 능력", "근력·근지구력", 0.106, 3, 0.015, 13),
            ("신체 능력", "순발력", 0.419, 2, 0.060, 7),
            ("신체 능력", "민첩성", 0.422, 1, 0.060, 6),
            ("운동기능수준", "운동 학습력", 0.318, 2, 0.139, 2),
            ("운동기능수준", "운동 감각(운동 센스)", 0.545, 1, 0.238, 1),
            ("운동기능수준", "전술 실행력", 0.137, 3, 0.060, 8),
            ("운동 경험", "학원 수강 경력", 0.350, 1, 0.095, 3),
            ("운동 경험", "학교스포츠클럽활동 경력", 0.302, 3, 0.082, 5),
            ("운동 경험", "체육특기자 여부(종목 무관)", 0.348, 2, 0.094, 4),
            ("인지적 능력", "규칙 이해도", 0.199, 3, 0.017, 12),
            ("인지적 능력", "전술 이해도", 0.262, 2, 0.023, 10),
            ("인지적 능력", "상황 판단력", 0.539, 1, 0.047, 9),
            ("사회·정서적 특성", "리더십", 0.067, 5, 0.004, 18),
            ("사회·정서적 특성", "수업참여도", 0.303, 1, 0.019, 11),
            ("사회·정서적 특성", "협동심", 0.220, 2, 0.014, 14),
            ("사회·정서적 특성", "자신감", 0.205, 4, 0.013, 16),
            ("사회·정서적 특성", "도전의식/용기", 0.206, 3, 0.013, 15),
        ],
    },
    "티볼": {
        "main": {"운동기능수준": (0.400, 1), "운동 경험": (0.249, 2), "인지적 능력": (0.175, 3),
                 "신체 능력": (0.109, 4), "사회·정서적 특성": (0.068, 5)},
        "main_cr": 0.029,
        "sub_cr": {"신체 능력": 0.044, "운동기능수준": 0.007, "운동 경험": 0.002,
                   "인지적 능력": 0.000, "사회·정서적 특성": 0.006},
        "sub": [
            ("신체 능력", "심폐지구력", 0.045, 4, 0.005, 18),
            ("신체 능력", "근력·근지구력", 0.137, 3, 0.015, 13),
            ("신체 능력", "순발력", 0.507, 1, 0.055, 8),
            ("신체 능력", "민첩성", 0.310, 2, 0.034, 11),
            ("운동기능수준", "운동 학습력", 0.200, 3, 0.080, 6),
            ("운동기능수준", "운동 감각(운동 센스)", 0.549, 1, 0.219, 1),
            ("운동기능수준", "전술 실행력", 0.251, 2, 0.100, 2),
            ("운동 경험", "학원 수강 경력", 0.378, 2, 0.094, 5),
            ("운동 경험", "학교스포츠클럽활동 경력", 0.396, 1, 0.099, 4),
            ("운동 경험", "체육특기자 여부(종목 무관)", 0.226, 3, 0.056, 7),
            ("인지적 능력", "규칙 이해도", 0.199, 3, 0.035, 10),
            ("인지적 능력", "전술 이해도", 0.233, 2, 0.041, 9),
            ("인지적 능력", "상황 판단력", 0.568, 1, 0.099, 3),
            ("사회·정서적 특성", "리더십", 0.137, 5, 0.009, 17),
            ("사회·정서적 특성", "수업참여도", 0.389, 1, 0.026, 12),
            ("사회·정서적 특성", "협동심", 0.168, 2, 0.011, 14),
            ("사회·정서적 특성", "자신감", 0.154, 3, 0.010, 15),
            ("사회·정서적 특성", "도전의식/용기", 0.152, 4, 0.010, 16),
        ],
    },
}
SPORTS = list(AHP.keys())

# 리더십은 '주장 사전 배정' 기능으로 대체하므로 편성 요인에서 제외
EXCLUDED_FACTORS = {"리더십"}

# ════════════════════════════════════════════════════════════════
# 2. 측정 척도 및 학생 설문 문항
# ════════════════════════════════════════════════════════════════
#  kind
#   paps   : PAPS 등급 1~5 (1등급이 가장 우수) → 역채점 점수 = 6 - 등급
#   years  : 경력 5점 (1 없음 ~ 5) — 학생 응답
#   binary : 예(1) / 아니오(0) — 학생 응답
#   likert : 5점 리커트(1 전혀 그렇지 않다 ~ 5 매우 그렇다), 문항 평균 — 학생 응답, 교사 수정 가능
#   count5 : 4개 문항 중 '알고 있다/할 수 있다'에 체크한 개수 → COUNT_TO_SCORE로 1~5점
KIND = {
    "심폐지구력": "paps", "근력·근지구력": "paps", "순발력": "paps",
    "민첩성": "likert",   # PAPS에 민첩성 영역 없음 → 자기평가 1문항 (논문 표 Ⅳ-27)
    "학원 수강 경력": "years", "학교스포츠클럽활동 경력": "years",
    "체육특기자 여부(종목 무관)": "binary",
    "규칙 이해도": "count5", "전술 이해도": "count5", "상황 판단력": "count5",
}

# 체크 개수(0~4) → 점수(1~5)
COUNT_TO_SCORE = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}


def kind(f):
    return KIND.get(f, "likert")


PAPS_OPTIONS = ["1등급", "2등급", "3등급", "4등급", "5등급"]
# 운동 경험(학원 수강·학교스포츠클럽 활동) 기간 5구간
YEARS_LABELS = {1: "없음", 2: "1년 이내", 3: "1년 초과 ~ 2년 이내", 4: "2년 초과 ~ 3년 이내", 5: "3년 초과"}
LIKERT_LABELS = {1: "전혀 그렇지 않다", 2: "그렇지 않다", 3: "보통이다", 4: "그렇다", 5: "매우 그렇다"}

# 경험 문항의 종목 표현 (티볼은 야구·연식야구 등 유사 종목 경험 포함)
EXP_OBJ = {"농구": "농구를", "배구": "배구를", "티볼": "티볼·야구·연식야구를"}
EXP_NAME = {"농구": "농구", "배구": "배구", "티볼": "티볼·야구·연식야구"}

COMMON_ITEMS = {
    "운동 학습력": ["나는 새로운 운동 동작을 배우면 빠르게 익히는 편이다."],
    "운동 감각(운동 센스)": ["나는 운동 센스가 좋다는 말을 자주 듣고, 실제로도 몸을 쓰는 감각이 좋은 편이다."],
    "전술 실행력": ["나는 {s} 경기에서 알고 있는 전술(작전)을 실제 경기 중에 해낼 수 있다."],
    "민첩성": ["나는 방향을 빠르게 바꾸거나 재빠르게 움직이는 편이다."],
    "수업참여도": ["나는 체육수업에 적극적으로 참여한다."],
    "협동심": ["나는 팀 활동에서 친구들과 소통하고 서로의 역할을 존중하며 협력한다."],
    "자신감": ["나는 운동할 때 내 능력을 믿고 위축되지 않고 적극적으로 행동한다."],
    "도전의식/용기": ["나는 새로운 시도나 어려운 상황 앞에서 두려워하지 않고 도전한다."],
}

# 규칙 이해도 · 전술 이해도 · 상황 판단력 : 종목별 4문항(체크 개수 0~4 → 1~5점) (확정본)
SPORT_ITEMS = {
    "농구": {
        "규칙 이해도": [
            "나는 공이 경기장 밖으로 나갔을 때 경기가 어떻게 다시 시작되는지 알고 있다.",
            "나는 어떤 신체 접촉이 파울이 되는지 알고 있다.",
            "나는 트래블링과 더블 드리블이 무엇인지 설명할 수 있다.",
            "나는 슛 동작 중 파울을 당했을 때 받는 자유투 개수를 알고 있다.",
        ],
        "전술 이해도": [
            "나는 속공과 지공의 차이를 설명할 수 있다.",
            "나는 대인 방어와 지역 방어의 차이를 설명할 수 있다.",
            "나는 스크린 플레이가 무엇인지 설명할 수 있다.",
            "나는 지공 상황에서 쓰는 공격 전술을 3가지 이상 알고 있다.",
        ],
        "상황 판단력": [
            "나는 공을 잡았을 때 패스, 드리블, 슛 중 무엇을 할지 결정할 수 있다.",
            "나는 경기 중 동료를 찾아 패스할 수 있다.",
            "나는 공격에서 수비로 바뀌는 순간 내가 막아야 할 상대나 위치를 바로 찾을 수 있다.",
            "나는 수비가 떨어져 있으면 슛을, 바짝 붙어 있으면 드리블을 선택할 수 있다.",
        ],
    },
    "배구": {
        "규칙 이해도": [
            "나는 한 팀이 정해진 횟수 안에 공을 넘겨야 한다는 규칙을 알고 있다.",
            "나는 한 선수가 공을 연속으로 두 번 칠 수 없다는 규칙을 알고 있다.",
            "나는 공의 인(in), 아웃(out) 판정 규칙을 알고 있다.",
            "나는 네트 터치, 터치아웃과 같은 실점 규칙을 알고 있다.",
        ],
        "전술 이해도": [
            "나는 상대의 공격을 받을 때 우리 팀의 기본 대형과 내 위치를 알고 있다.",
            "나는 리시브, 토스, 공격으로 이어지는 공격 흐름을 알고 있다.",
            "나는 세터, 공격수 등 포지션별 역할을 알고 있다.",
            "나는 우리 팀이 블로킹할 때 주변 동료가 커버하러 가야 한다는 것을 알고 있다.",
        ],
        "상황 판단력": [
            "나는 경기 중 상대의 빈 공간으로 공을 보낼 수 있다.",
            "나는 나와 동료 사이로 공이 올 때, 누가 받을지 빠르게 판단하여 '콜'할 수 있다.",
            "나는 경기 중 상대의 공격 방향을 예측하여 미리 움직일 수 있다.",
            "나는 동료가 공을 받지 못했을 때 곧바로 커버하러 움직일 수 있다.",
        ],
    },
    "티볼": {
        "규칙 이해도": [
            "나는 타자가 스트라이크가 되는 경우를 알고 있다.",
            "나는 플라이아웃, 포스아웃, 태그아웃 등 아웃이 되는 경우를 알고 있다.",
            "나는 페어볼과 파울볼의 구분 기준을 알고 있다.",
            "나는 주자가 진루할 때 지켜야 할 규칙을 알고 있다.",
        ],
        "전술 이해도": [
            "나는 주자가 있을 때 어느 베이스로 공을 던져야 하는지 알고 있다.",
            "나는 내야수, 외야수 등 포지션별 역할을 알고 있다.",
            "나는 수비수의 위치를 보고 빈 공간으로 공을 보내는 타격 방법을 알고 있다.",
            "나는 백업 플레이와 중계 플레이에 대해 알고 있다.",
        ],
        "상황 판단력": [
            "나는 타구가 날아오는 방향과 거리를 빠르게 판단하여 수비할 수 있다.",
            "나는 공을 잡았을 때 어느 베이스로 던질지 빠르게 결정할 수 있다.",
            "나는 주자일 때 다음 베이스로 뛸지 멈출지 상황에 맞게 판단할 수 있다.",
            "나는 아웃카운트와 주자 상황을 고려하여 수비 플레이를 선택할 수 있다.",
        ],
    },
}

GROUPS = [  # 학생 화면 구성 순서
    (":material/fitness_center: 체력 (PAPS)", ["심폐지구력", "근력·근지구력", "순발력", "민첩성"]),
    (":material/military_tech: 운동 경험", ["학원 수강 경력", "학교스포츠클럽활동 경력", "체육특기자 여부(종목 무관)"]),
    (":material/psychology: 경기 이해", ["규칙 이해도", "전술 이해도", "상황 판단력"]),
    (":material/bolt: 운동 능력", ["운동 학습력", "운동 감각(운동 센스)", "전술 실행력"]),
    (":material/handshake: 수업 태도", ["수업참여도", "협동심", "자신감", "도전의식/용기"]),
]


def items(sport, f):
    """요인 f의 학생 문항 목록 [(문항ID, 문항)]"""
    k = kind(f)
    if k == "paps":
        return [(f"{f}#1", f"최근 PAPS(학생건강체력평가) {f} 등급은?")]
    if k == "years":
        if f.startswith("학원"):
            return [(f"{f}#1", f"{EXP_OBJ[sport]} 학원에서 배운 기간은?")]
        return [(f"{f}#1", f"{EXP_NAME[sport]} 학교스포츠클럽 활동을 한 기간은?")]
    if k == "binary":
        return [(f"{f}#1", "나는 체육 특기자(종목 무관)이다.")]
    lst = SPORT_ITEMS.get(sport, {}).get(f) or COMMON_ITEMS.get(f, [f"나는 {f}이(가) 높은 편이다."])
    return [(f"{f}#{i + 1}", q.format(s=sport)) for i, q in enumerate(lst)]


def questionnaire_table(sport, factors=None):
    rows = []
    for g, fs in GROUPS:
        for f in fs:
            if f in EXCLUDED_FACTORS or (factors is not None and f not in factors):
                continue
            for iid, q in items(sport, f):
                scale = {"paps": "1~5등급(역채점)",
                         "years": " / ".join(f"{k} {v}" for k, v in YEARS_LABELS.items()),
                         "binary": "예(1) / 아니오(0)",
                         "likert": "전혀 그렇지 않다 1 ~ 매우 그렇다 5",
                         "count5": "체크(알고 있음/할 수 있음) · 4문항 중 체크 개수 → "
                                   + ", ".join(f"{c}개={v}점" for c, v in COUNT_TO_SCORE.items())}[kind(f)]
                rows.append({"영역": g.split(" ", 1)[1], "요인": f, "문항번호": iid.split("#")[1],
                             "문항": q, "응답 척도": scale})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# 3. AHP 관련 함수
# ════════════════════════════════════════════════════════════════
def sub_table(sport, include_excluded=True):
    df = pd.DataFrame(AHP[sport]["sub"],
                      columns=["대분류", "하위요인", "하위요인 가중치", "하위요인 순위", "종합 가중치", "종합 순위"])
    if not include_excluded:
        df = df[~df["하위요인"].isin(EXCLUDED_FACTORS)]
    return df.sort_values("종합 순위").reset_index(drop=True)


def main_table(sport):
    m = AHP[sport]["main"]
    df = pd.DataFrame([(k, v[0], v[1]) for k, v in m.items()], columns=["대분류 요인", "가중치", "순위"])
    return df.sort_values("순위").reset_index(drop=True)


def top_factors(sport, n=5):
    return sub_table(sport, include_excluded=False).head(n)["하위요인"].tolist()


def applied_weights(sport, factors):
    """선택 요인의 적용 가중치 = AHP 종합 가중치를 선택 요인 내에서 합계 1로 재정규화"""
    if not factors:
        return {}
    g = dict(zip(sub_table(sport)["하위요인"], sub_table(sport)["종합 가중치"]))
    tot = sum(g[f] for f in factors)
    return {f: g[f] / tot for f in factors}


# ════════════════════════════════════════════════════════════════
# 4. 점수 변환
#    원점수 → 방향 통일(높을수록 우수, PAPS 역채점) → 학급 표준편차로 나눔(계획서 '데이터의 정의')
# ════════════════════════════════════════════════════════════════
TEXT_MAP_SCALE = {"상": 5, "중": 3, "하": 1}
TEXT_MAP_BIN = {"예": 1, "있음": 1, "o": 1, "y": 1, "yes": 1, "true": 1, "해당": 1,
                "아니오": 0, "아니요": 0, "없음": 0, "x": 0, "n": 0, "no": 0, "false": 0, "미해당": 0}
GENDER_COL = "성별(남0/여1)"


def parse_value(v, f):
    """원점수 해석 (paps: 등급 1~5, years/likert: 1~5, binary: 0/1). 해석 불가·빈칸 → NaN"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    k = kind(f)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "nan", "none"):
            return np.nan
        if k == "binary" and s in TEXT_MAP_BIN:
            return float(TEXT_MAP_BIN[s])
        if k in ("likert", "years", "count5") and s in TEXT_MAP_SCALE:
            return float(TEXT_MAP_SCALE[s])
        m = re.search(r"\d+(\.\d+)?", s)
        if not m:
            return np.nan
        v = m.group()
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    if k == "binary":
        return 1.0 if v >= 1 else 0.0
    return float(min(5, max(1, v)))


def direction_score(v, f):
    """높을수록 우수하도록 방향 통일 (PAPS 등급 역채점)"""
    return 6 - v if kind(f) == "paps" else v


def display_score01(v, f):
    """표시용 0~1 (가능 범위 기준)"""
    if kind(f) == "binary":
        return v
    return (v - 1) / 4


def parse_gender(v):
    s = str(v).strip().lower()
    return 1 if s in ("1", "1.0", "여", "여자", "f", "female") else 0


def build_score_matrix(df, factors):
    """
    returns raw(입력값), score(방향 통일), z(표준편차로 나눈 값: MILP 투입), warnings
    결측(교사 입력 빈칸) → 해당 요인 학급 평균 대체
    """
    raw = pd.DataFrame(index=df.index)
    for f in factors:
        col = df[f] if f in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        raw[f] = [parse_value(v, f) for v in col]
    score = raw.apply(lambda c: c.map(lambda v: direction_score(v, c.name) if not np.isnan(v) else np.nan))
    z = pd.DataFrame(index=df.index)
    warnings = []
    for f in factors:
        miss = int(score[f].isna().sum())
        if miss == len(score):
            warnings.append(f"'{f}' 값이 모두 비어 있어 편성에 반영되지 않습니다.")
            score[f] = 0.0
            z[f] = 0.0
            continue
        if miss:
            names = df.loc[score[f].isna(), "이름"].astype(str).tolist()
            warnings.append(f"'{f}' 결측 {miss}명({', '.join(names[:5])}{' 외' if miss > 5 else ''}) → 학급 평균으로 대체")
            score[f] = score[f].fillna(score[f].mean())
        sd = score[f].std(ddof=0)
        if sd == 0:
            warnings.append(f"'{f}' 모든 학생이 같은 값이라 편성에 영향이 없습니다.")
            z[f] = 0.0
        else:
            z[f] = score[f] / sd
    return raw, score, z, warnings


# ════════════════════════════════════════════════════════════════
# 5. MILP 팀 편성 (Rubin & Bai, 2015 방식)
# ════════════════════════════════════════════════════════════════
PULP_FIX_CMD = 'python -m pip install "pulp==2.9.0"'


def pulp_compatible():
    """이 프로그램은 PuLP 2.x(고전 API)를 사용. 3.x 이상은 사용법이 달라 동작하지 않음"""
    import inspect
    try:
        ver = getattr(pulp, "__version__", "?")
        ok = int(str(ver).split(".")[0]) < 3
        try:
            ok = ok and "cat" in inspect.signature(LpVariable.__init__).parameters
        except (TypeError, ValueError):
            pass
        return ok, ver
    except Exception:
        return True, "?"


def run_milp(names, genders, score, weights, n_teams, min_female,
             same_pairs=(), diff_pairs=(), captains=(), time_limit=60):
    """
    [의사결정 변수]  x[i][t] ∈ {0,1}
    [목적함수]      minimize Σ_t Σ_a w_a (d⁺[t][a] + d⁻[t][a])
        d⁺[t][a] - d⁻[t][a] = (Σ_i s_ia·x_it) / n_t - μ_a
        n_t : 팀 t의 인원(상수). 팀 인원은 n//T 또는 n//T+1 로만 가능하므로
              앞의 r(= n mod T)개 팀을 n//T+1명, 나머지를 n//T명으로 미리 정함
              (팀 번호는 교환 가능하므로 일반성 손실 없음)
        → 분모가 상수가 되어 Rubin & Bai(2015)의 비선형 모형(팀 평균 기준)과 수학적으로 동일
        w_a : AHP 종합 가중치(선택 요인 내 재정규화), s_ia : 학급 표준편차로 나눈 점수
    [제약조건]
        강한 제약  : 1인 1팀, 팀 인원(차이 최대 1명), 팀당 최소 여학생 수
        부수적 제약: 지정한 두 학생 결합(같은 팀)·분리(다른 팀) — 여러 쌍 지정 가능, 주장 팀별 최대 1명
            결합 (i, j): x[i][t] = x[j][t]      ∀t
            분리 (i, j): x[i][t] + x[j][t] ≤ 1  ∀t
    """
    score = np.asarray(score, dtype=float)
    n, A = score.shape
    T = n_teams
    mu = score.mean(axis=0)
    base, r = divmod(n, T)
    size = [base + 1 if t < r else base for t in range(T)]

    prob = LpProblem("TeamAssignment_MILP", LpMinimize)
    x = [[LpVariable(f"x_{i}_{t}", cat="Binary") for t in range(T)] for i in range(n)]
    dp = [[LpVariable(f"dp_{t}_{a}", lowBound=0) for a in range(A)] for t in range(T)]
    dm = [[LpVariable(f"dm_{t}_{a}", lowBound=0) for a in range(A)] for t in range(T)]

    prob += lpSum(weights[a] * (dp[t][a] + dm[t][a]) for t in range(T) for a in range(A))

    for i in range(n):
        prob += lpSum(x[i][t] for t in range(T)) == 1
    for t in range(T):
        prob += lpSum(x[i][t] for i in range(n)) == size[t]
        for a in range(A):
            prob += dp[t][a] - dm[t][a] == \
                lpSum((float(score[i][a]) / size[t]) * x[i][t] for i in range(n)) - float(mu[a])
        if min_female > 0:
            prob += lpSum(x[i][t] for i in range(n) if genders[i] == 1) >= min_female

    idx = {nm: i for i, nm in enumerate(names)}
    for a, b in same_pairs:
        if a in idx and b in idx:
            for t in range(T):
                prob += x[idx[a]][t] == x[idx[b]][t]
    for a, b in diff_pairs:
        if a in idx and b in idx:
            for t in range(T):
                prob += x[idx[a]][t] + x[idx[b]][t] <= 1
    caps = [idx[c] for c in captains if c in idx]
    if len(caps) >= 2:
        for t in range(T):
            prob += lpSum(x[i][t] for i in caps) <= 1

    # 대칭성 제거(계산 속도 향상, 해의 질과 무관): 첫 학생은 '큰 팀 중 첫 팀' 또는 '작은 팀 중 첫 팀'
    first = [0] + ([r] if 0 < r < T else [])
    prob += lpSum(x[0][t] for t in first) == 1

    solver = _make_solver(time_limit)
    if solver is not None:
        prob.solve(solver)
    else:
        prob.solve()
    outcome = _solve_outcome(prob)
    if outcome != "optimal":
        return {"error": outcome}
    team_of = {}
    for i in range(n):
        for t in range(T):
            if value(x[i][t]) is not None and value(x[i][t]) > 0.5:
                team_of[i] = t
    if len(team_of) != n:
        return {"error": "infeasible"}
    # 팀 번호 정리: 주장 선택 순서대로 팀 1, 2, … (나머지 팀은 그 뒤)
    order = []
    for i in caps:
        if team_of[i] not in order:
            order.append(team_of[i])
    order += [t for t in range(T) if t not in order]
    relabel = {t: k + 1 for k, t in enumerate(order)}
    assign = {names[i]: relabel[t] for i, t in team_of.items()}
    return {"assign": assign, "objective": value(prob.objective)}


def clean_pairs(df_pairs):
    """짝 입력표 → [(학생A, 학생B)] (빈칸·자기 자신·중복 쌍 제거)"""
    out, seen = [], set()
    if df_pairs is None or len(df_pairs) == 0:
        return out
    for a, b in df_pairs[["학생 A", "학생 B"]].itertuples(index=False):
        a = "" if a is None or (isinstance(a, float) and np.isnan(a)) else str(a).strip()
        b = "" if b is None or (isinstance(b, float) and np.isnan(b)) else str(b).strip()
        if not a or not b or a == b:
            continue
        key = frozenset((a, b))
        if key not in seen:
            seen.add(key)
            out.append((a, b))
    return out


def check_pairs(names, n_teams, same_pairs, diff_pairs, captains):
    """결합·분리·주장 조건 사이의 모순 점검 → 오류 메시지 목록"""
    errs = []
    nameset = set(names)
    gone = sorted({s for p in list(same_pairs) + list(diff_pairs) for s in p if s not in nameset})
    if gone:
        errs.append(f"학생 명단에 없는 이름이 짝에 지정되어 있습니다: {', '.join(gone)}")
    # 결합 쌍을 이어서 '반드시 같은 팀'이 되는 묶음 구성 (예: A-B, B-C → A·B·C)
    parent = {s: s for s in nameset}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    for a, b in same_pairs:
        if a in nameset and b in nameset:
            parent[find(a)] = find(b)
    groups = {}
    for s in nameset:
        groups.setdefault(find(s), []).append(s)
    cap_size = -(-len(names) // n_teams)
    for g in groups.values():
        if len(g) > cap_size:
            errs.append(f"같은 팀으로 이어진 학생({', '.join(sorted(g))})이 {len(g)}명으로 팀 정원({cap_size}명)을 초과합니다.")
    for a, b in diff_pairs:
        if a in nameset and b in nameset and find(a) == find(b):
            errs.append(f"{a}·{b}는 결합 조건으로 같은 팀이 되므로 분리할 수 없습니다.")
    caps = [c for c in captains if c in nameset]
    for g in groups.values():
        cg = [c for c in caps if c in g]
        if len(cg) >= 2:
            errs.append(f"주장 {', '.join(cg)}이(가) 결합 조건으로 같은 팀이 되어 주장 조건과 충돌합니다.")
    return errs


def team_label(t):
    """팀 번호(1, 2, 3 …) → 화면 표시용 팀 이름(A, B, C …)"""
    return chr(64 + int(t))


def summarize(names, genders, raw, score, weights_d, assign, n_teams, captains=()):
    factors = list(weights_d.keys())
    w = np.array([weights_d[f] for f in factors])
    stu = pd.DataFrame({"이름": names, GENDER_COL: genders})
    stu["주장"] = ["★" if nm in captains else "" for nm in names]
    for f in factors:
        stu[f] = raw[f].values
    disp = np.column_stack([[display_score01(v, f) for v in score[f]] for f in factors])
    stu["종합 점수(0~100)"] = (disp @ w * 100).round(1)
    stu["배정 팀"] = stu["이름"].map(assign)

    rows = []
    for t in range(1, n_teams + 1):
        m = stu[stu["배정 팀"] == t]
        cap = m.loc[m["주장"] == "★", "이름"].tolist()
        r = {"팀": f"팀 {team_label(t)}", "주장": ", ".join(cap), "인원": len(m), "여학생": int((m[GENDER_COL] == 1).sum())}
        for f in factors:
            if kind(f) == "binary":
                r[f] = f"{int(np.nansum(m[f]))}명"
            else:
                r[f] = round(float(np.nanmean(m[f])), 2) if m[f].notna().any() else None
        r["종합 점수 평균"] = round(m["종합 점수(0~100)"].mean(), 1)
        rows.append(r)
    return stu.sort_values(["배정 팀", "주장", "이름"], ascending=[True, False, True]).reset_index(drop=True), \
        pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# 6. 학생 참여용 저장소 (SQLite)
# ════════════════════════════════════════════════════════════════
DB_PATH = os.environ.get("TEAM_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_data.db"))


def _db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS sessions(
        code TEXT PRIMARY KEY, title TEXT, sport TEXT, factors TEXT,
        is_open INTEGER, results TEXT, created TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS submissions(
        code TEXT, name TEXT, gender INTEGER, answers TEXT, ts TEXT,
        PRIMARY KEY(code, name))""")
    return con


def create_session(title, sport, student_factors):
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    with _db() as con:
        while True:
            code = "".join(random.choice(chars) for _ in range(6))
            if not con.execute("SELECT 1 FROM sessions WHERE code=?", (code,)).fetchone():
                break
        con.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?)",
                    (code, title, sport, json.dumps(student_factors, ensure_ascii=False), 1, None,
                     datetime.now().isoformat(timespec="seconds")))
    return code


# 보관 기간: 참여코드(세션)와 학생 응답은 생성 후 약 6개월(183일)이 지나면 자동 삭제
RETENTION_DAYS = 183


def purge_old_sessions():
    """보관 기간이 지난 세션과 그 세션의 학생 응답을 삭제. 삭제한 세션 수를 반환"""
    cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).isoformat(timespec="seconds")
    with _db() as con:
        old = [r[0] for r in con.execute("SELECT code FROM sessions WHERE created < ?", (cutoff,)).fetchall()]
        for c in old:
            con.execute("DELETE FROM submissions WHERE code=?", (c,))
            con.execute("DELETE FROM sessions WHERE code=?", (c,))
    return len(old)


def get_session(code):
    with _db() as con:
        r = con.execute("SELECT code,title,sport,factors,is_open,results FROM sessions WHERE code=?",
                        (code.strip().upper(),)).fetchone()
    if not r:
        return None
    return {"code": r[0], "title": r[1], "sport": r[2], "factors": json.loads(r[3]),
            "is_open": bool(r[4]), "results": json.loads(r[5]) if r[5] else None}


def set_session_open(code, is_open):
    with _db() as con:
        con.execute("UPDATE sessions SET is_open=? WHERE code=?", (int(is_open), code))


def publish_results(code, payload):
    with _db() as con:
        con.execute("UPDATE sessions SET results=? WHERE code=?",
                    (json.dumps(payload, ensure_ascii=False) if payload else None, code))


def save_submission(code, name, gender, item_answers, factor_values):
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO submissions VALUES(?,?,?,?,?)",
                    (code, name, gender,
                     json.dumps({"items": item_answers, "factors": factor_values}, ensure_ascii=False),
                     datetime.now().isoformat(timespec="seconds")))


def delete_submission(code, name):
    with _db() as con:
        con.execute("DELETE FROM submissions WHERE code=? AND name=?", (code, name))


def _rows(code):
    with _db() as con:
        return con.execute("SELECT name,gender,answers,ts FROM submissions WHERE code=? ORDER BY ts",
                           (code,)).fetchall()


# ── 교사 확인·수정 값 저장 ──
#   answers JSON = {"items": 학생 문항 원응답, "factors": 학생 응답 요인 점수,
#                   "teacher": 교사가 수정한 요인 점수, "teacher_gender": 교사가 수정한 성별,
#                   "by_teacher": 교사가 직접 추가한 학생 여부, "hidden": 교사 표에서 삭제한 학생 여부}
#   학생 원응답(items·factors)은 그대로 보존하고, 교사 수정값은 별도로 저장하여 우선 적용한다.
def _update_answers(code, name, fn):
    with _db() as con:
        r = con.execute("SELECT answers FROM submissions WHERE code=? AND name=?", (code, name)).fetchone()
        if not r:
            return
        ans = json.loads(r[0])
        fn(ans)
        con.execute("UPDATE submissions SET answers=? WHERE code=? AND name=?",
                    (json.dumps(ans, ensure_ascii=False), code, name))


def save_teacher_edits(code, name, values, gender=None):
    def fn(ans):
        ans.setdefault("teacher", {}).update(values)
        if gender is not None:
            ans["teacher_gender"] = gender
    _update_answers(code, name, fn)


def add_teacher_row(code, name, gender, values):
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO submissions VALUES(?,?,?,?,?)",
                    (code, name, gender,
                     json.dumps({"items": {}, "factors": {}, "teacher": values, "by_teacher": True},
                                ensure_ascii=False),
                     datetime.now().isoformat(timespec="seconds")))


def hide_submission(code, name):
    _update_answers(code, name, lambda ans: ans.update(hidden=True))


# ── 학생 데이터 파일(선생님 PC 보관용) 저장·복원 ──
#   시트 '세션': 참여코드·세션 정보 / 시트 '응답': DB 원자료(학생 응답·교사 수정값 포함) / 시트 '학생 데이터': 보기용
BACKUP_SHEET = "복원용"            # 세션 정보(첫 행) + 학생 원자료(응답·교사 수정값)를 한 시트에
SESSION_ROW = "__session__"


def write_backup_sheets(xw, code):
    """복원용 시트 1개: 첫 행에 세션 정보, 다음 행부터 학생별 원자료(JSON)"""
    with _db() as con:
        s = con.execute("SELECT code,title,sport,factors,is_open,results,created FROM sessions WHERE code=?",
                        (code,)).fetchone()
        rows = con.execute("SELECT name,gender,answers,ts FROM submissions WHERE code=? ORDER BY ts",
                           (code,)).fetchall()
    sess = dict(zip(["code", "title", "sport", "factors", "is_open", "results", "created"], s))
    data = [(SESSION_ROW, None, json.dumps(sess, ensure_ascii=False), None)] + list(rows)
    pd.DataFrame(data, columns=["name", "gender", "answers", "ts"]).to_excel(xw, sheet_name=BACKUP_SHEET, index=False)


def write_student_sheets(xw, code):
    """학생 데이터: 설문한 모든 요인의 점수(교사 수정 반영, 표에서 지운 학생 제외)"""
    sess = get_session(code)
    if sess:
        all_f = set(sub_table(sess["sport"], include_excluded=False)["하위요인"])
        name = "전체 문항 학생 데이터" if all_f <= set(sess["factors"]) else "일부 문항 학생 데이터"
        load_submissions(code, sess["factors"]).drop(columns=["제출시각"]) \
            .to_excel(xw, sheet_name=name, index=False)


def export_session_backup(code):
    """학생 데이터 저장 파일: ① 학생 데이터 ② 복원용"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        write_student_sheets(xw, code)
        write_backup_sheets(xw, code)
    return buf.getvalue()


def restore_session_backup(file, code):
    """저장해 둔 파일로 세션 복원. 입력한 참여코드와 파일의 코드가 같아야 함 → (성공 여부, 메시지)"""
    try:
        sheets = pd.read_excel(file, sheet_name=None, dtype=str)
        if BACKUP_SHEET in sheets:                                  # 현재 형식
            b = sheets[BACKUP_SHEET].fillna("")
            s = json.loads(b.iloc[0]["answers"])
            resp = b.iloc[1:]
        else:                                                       # 이전 형식(세션·응답 시트)
            s = sheets["세션"].fillna("").iloc[0].to_dict()
            resp = sheets["응답"].fillna("")
    except Exception:
        return False, "복원할 수 있는 파일이 아닙니다. '학생 데이터 저장'으로 받은 파일을 올려 주세요."
    if str(s["code"]).strip().upper() != code:
        return False, "입력한 참여코드와 파일의 참여코드가 다릅니다."
    results = s["results"] if isinstance(s["results"], str) and s["results"].strip() else None
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO sessions VALUES(?,?,?,?,?,?,?)",
                    (code, s["title"], s["sport"], s["factors"], int(float(s["is_open"])), results,
                     datetime.now().isoformat(timespec="seconds")))
        con.execute("DELETE FROM submissions WHERE code=?", (code,))
        for _, r in resp.iterrows():
            if not str(r["name"]).strip():
                continue
            g = None if str(r["gender"]).strip() in ("", "nan", "None") else int(float(r["gender"]))
            con.execute("INSERT INTO submissions VALUES(?,?,?,?,?)",
                        (code, str(r["name"]), g, r["answers"] or "{}", r["ts"]))
    return True, f"참여코드 {code}의 학생 데이터를 복원했습니다."


def load_submissions(code, factors):
    """요인 단위 점수표 (교사 확인·수정용) — 교사 수정값이 있으면 학생 응답보다 우선"""
    data = []
    for name, g, ans, ts in _rows(code):
        a = json.loads(ans)
        if a.get("hidden"):
            continue
        fv, tv = a.get("factors", {}), a.get("teacher", {})
        r = {"이름": name, GENDER_COL: a.get("teacher_gender", g)}
        for f in factors:
            r[f] = tv[f] if f in tv else fv.get(f)
        r["제출시각"] = ts
        data.append(r)
    return pd.DataFrame(data, columns=["이름", GENDER_COL] + list(factors) + ["제출시각"])


def _same(a, b):
    na = a is None or (isinstance(a, float) and np.isnan(a))
    nb = b is None or (isinstance(b, float) and np.isnan(b))
    if na or nb:
        return na and nb
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def _plain(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        fv = float(v)
        return int(fv) if fv.is_integer() else fv
    except (TypeError, ValueError):
        return None


def sync_teacher_table(code, before, after, factors):
    """교사 표(after)와 DB 값(before)을 비교하여 변경 사항을 DB에 저장. 변경이 있으면 True"""
    changed = False
    prev = {str(r["이름"]): r for _, r in before.iterrows()}
    seen = set()
    for _, r in after.iterrows():
        name = "" if r.get("이름") is None or (isinstance(r.get("이름"), float) and np.isnan(r.get("이름"))) \
            else str(r["이름"]).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        if name in prev:
            old = prev[name]
            edits = {f: _plain(r[f]) for f in factors if not _same(r[f], old[f])}
            g = None if _same(r[GENDER_COL], old[GENDER_COL]) else _plain(r[GENDER_COL])
            if edits or g is not None:
                save_teacher_edits(code, name, edits, g)
                changed = True
        else:
            add_teacher_row(code, name, _plain(r[GENDER_COL]), {f: _plain(r[f]) for f in factors})
            changed = True
    for name in prev:
        if name not in seen:
            hide_submission(code, name)
            changed = True
    return changed


def load_item_responses(code, sport):
    """문항 단위 원응답 (신뢰도 분석 등 연구용) — 학생이 직접 응답한 자료만"""
    data = []
    for name, g, ans, ts in _rows(code):
        if json.loads(ans).get("by_teacher"):
            continue
        it = json.loads(ans).get("items", {})
        r = {"이름": name, "성별": "여" if g == 1 else "남", "제출시각": ts}
        r.update(it)
        data.append(r)
    return pd.DataFrame(data)


# ════════════════════════════════════════════════════════════════
# 7. 화면
# ════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
/* 단색 선 아이콘 (Material Symbols) — HTML 제목 안에서 사용 */
.mi { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; font-size: 1.15em;
      line-height: 1; vertical-align: -0.18em; display: inline-block; letter-spacing: normal;
      text-transform: none; white-space: nowrap; -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; }

/* ── 전체 글씨 크기 ── */
html { font-size: 17.5px; }

/* ── 본문 폭 1440px 제한 · 여백 ── */
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1440px; padding-top: 2.5rem; padding-bottom: 4rem; margin: 0 auto; }

/* ── 사이드바: 폭 넓게 · 글씨 크게 ── */
section[data-testid="stSidebar"][aria-expanded="true"] { width: 380px !important; min-width: 380px !important; }
/* 사이드바를 접으면 본문을 화면 폭에 맞춰 넓게 */
section[data-testid="stSidebar"][aria-expanded="false"] ~ section .block-container,
section[data-testid="stSidebar"][aria-expanded="false"] ~ section [data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"]:has(section[data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"]:has(section[data-testid="stSidebar"][aria-expanded="false"]) .block-container {
    max-width: 100%; padding-left: 3rem; padding-right: 3rem; }
[data-testid="stSidebarUserContent"] { padding-top: 0.25rem; }
.side-title { font-size: 1.35rem; font-weight: 700; line-height: 1.45; margin: -0.75rem 0 0.9rem; }
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: 1.08rem; }
section[data-testid="stSidebar"] h3 { font-size: 1.35rem; }

/* ── 세로 간격 통일 ── */
[data-testid="stVerticalBlock"] { gap: 0.75rem; }
.section-header { font-size: 1.1rem; font-weight: 700; color: #1a3a5c;
                  border-left: 4px solid #2e86de; padding-left: 10px; margin: 2rem 0 0.7rem; }

/* ── 입력창 너비 통일 ── */
[data-testid="stNumberInput"] { max-width: 260px; }
[data-testid="stTextInput"]   { max-width: 420px; }

/* ── 제목 ── */
.main-title { font-size: 1.9rem; font-weight: 700; color: #1a3a5c; margin-bottom: 0.2rem; }
.sub-title  { font-size: 0.95rem; color: #5a7a9a; margin-bottom: 1.5rem; }

/* ── 결과: 팀 카드 · 배지 ── */
.team-card { background: #f0f6ff; border-radius: 12px; padding: 1rem 1.2rem;
             margin-bottom: 0.8rem; border: 1px solid #c8dff8; }
.team-card h4 { color: #1a3a5c; margin: 0 0 0.5rem; font-size: 1.25rem; }
.team-card h4 small { font-size: 0.85rem; font-weight: 400; }
.badge { display:inline-block; background:#2e86de; color:white;
         border-radius:20px; padding:2px 10px; font-size:0.8rem; margin:2px; }
.badge-f { background: #e84393; }
.badge-c { outline: 2px solid #f6b93b; outline-offset: 1px; }

/* ── 결과 지표 ── */
.metric-box { text-align:center; background:#fff; border:1px solid #dde8f8;
              border-radius:10px; padding:0.8rem; margin-bottom: 1rem; }
.metric-box .val { font-size:1.4rem; font-weight:700; color:#2e86de; }
.metric-box .lbl { font-size:0.8rem; color:#888; }

/* ── 참여코드 ── */
.code-box { font-size:2.4rem; font-weight:700; letter-spacing:0.4rem; color:#1a3a5c;
            background:#f0f6ff; border:2px dashed #2e86de; border-radius:12px;
            text-align:center; padding:0.6rem; margin-bottom: 1rem; }
.q { font-weight:600; margin-top:0.6rem; }
/* 데이터 저장 버튼: 눈에 띄게 노란색 */
.st-key-save_result button, .st-key-save_student button {
    background: #FFE066 !important; border: 1px solid #F5C400 !important; color: #3d3200 !important; font-weight: 600; }
.st-key-save_result button:hover, .st-key-save_student button:hover { background: #FFD43B !important; }
.st-key-save_result { position: relative; top: 2.7rem; }       /* 팀 카드 바로 위에 오도록 */
.sub-head { font-size: 1rem; font-weight: 700; color: #1a3a5c; margin: 0.8rem 0 0.35rem; }
/* ■ 소제목 + 토글: 글자를 왼쪽, 스위치를 바로 오른쪽에 (키가 tg_ 로 시작하는 토글만) */
[class*="st-key-tg_"] { margin-top: 0.8rem; }
[class*="st-key-tg_"] label { flex-direction: row-reverse; justify-content: flex-end; gap: 0.6rem; }
[class*="st-key-tg_"] label p { font-size: 1rem; font-weight: 700; color: #1a3a5c; }
.cr-label { text-align: right; font-size: 0.85rem; color: #6b7280; margin: 0 0 0.2rem; line-height: 1.2; }
/* AHP 결과 표 */
.ahp-tbl { width: 100%; border-collapse: collapse; font-size: 0.95rem; background: #fff;
           border: 1px solid #e6e9ef; border-radius: 8px; overflow: hidden; }
.ahp-tbl th { background: #f8f9fb; color: #6b7280; font-weight: 600; text-align: left;
              padding: 0.5rem 0.7rem; border-bottom: 1px solid #e6e9ef; white-space: nowrap; }
.ahp-tbl td { padding: 0.5rem 0.7rem; border-bottom: 1px solid #eef0f4; color: #31333f; }
.ahp-tbl tr:last-child td { border-bottom: none; }
.ahp-tbl th + th, .ahp-tbl td + td { border-left: 1px solid #eef0f4; }
.ahp-tbl .c { text-align: center; } .ahp-tbl .r { text-align: right; }
.ahp-tbl th.c { text-align: center; } .ahp-tbl th.r { text-align: right; }
.ahp-tbl td { white-space: nowrap; height: 2.6rem; box-sizing: border-box; }
.ahp-tbl tr.top td { background: #dbeafe; font-weight: 600; }
.ahp-tbl tr.excl td { background: #e9ecef; }
/* 표 위에 마우스를 올리면 나오는 도구 막대(열 숨기기·CSV 다운로드·검색·전체 화면) 숨김 */
[data-testid="stElementToolbar"] { display: none !important; }
/* 단, 학생 데이터 표에서는 행 삭제(휴지통) 등을 위해 도구 막대를 표시 */
[class*="st-key-de_"] [data-testid="stElementToolbar"] {
    display: flex !important; left: 0 !important; right: auto !important; justify-content: flex-start !important; }
/* 도구 막대가 표 위쪽에 뜨므로, 위의 안내·버튼과 겹치지 않게 표 위에 공간 확보 */
[class*="st-key-de_"] { margin-top: 1.6rem; }

</style>
"""


INPUT_GUIDE_TABLE = """
| 항목 | 입력값 | 설명 |
|---|---|---|
| 성별 | 0 / 1 | 남 0, 여 1 |
| PAPS (심폐지구력·근력·근지구력·순발력) | 1\\~5 | 등급 그대로 입력 (1등급 = 1, 프로그램이 자동으로 역산) |
| 경력 (학원 수강·학교스포츠클럽 활동) | 1\\~5 | 없음(1), 1년 이내(2), 1\\~2년(3), 2\\~3년(4), 3년 초과(5) |
| 체육특기자 여부 | 0 / 1 | 아니오 0, 예 1 |
| 인지적 능력 (규칙 이해도·전술 이해도·상황 판단력) | 1\\~5 | 4문항 중 해당 개수 + 1 (0개 = 1점, 4개 = 5점) |
| 그 외 요인 (운동 감각, 협동심 등) | 1\\~5 | 5점 척도 (전혀 그렇지 않다 1 \\~ 매우 그렇다 5) |
"""


def input_guide(lead):
    """엑셀 업로드·직접 입력 공통 입력 안내: 한 줄 요약 + 펼쳐 보는 상세 기준"""
    st.caption(f"{lead} 성별(남0/여1) · PAPS 등급(1\\~5) · 경력(1\\~5) · 특기자(0/1) · 그 외 요인(1\\~5)")
    with st.expander("입력 기준 자세히 보기"):
        st.markdown(INPUT_GUIDE_TABLE)


def sub_header(text, mark=True):
    """단계 안의 소제목 — 모두 같은 크기·굵기 (mark=True면 앞에 ■)"""
    st.markdown(f'<div class="sub-head">{"■ " if mark else ""}{text}</div>', unsafe_allow_html=True)


def sub_header_toggle(text, key, value=False):
    """■ 소제목 바로 오른쪽에 토글 — 제목을 토글 이름표로 넣고 CSS로 '글자 → 스위치' 순서로 표시
    (하나의 요소이므로 화면 폭과 관계없이 같은 줄에 붙어 있음)"""
    return st.toggle(f"■ {text}", value=value, key=f"tg_{key}")


def cr_label(cr):
    """표 오른쪽 위 CR 표시"""
    st.markdown(f'<div class="cr-label">CR = {cr:.3f}</div>', unsafe_allow_html=True)


def header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def metric(col, val, lbl):
    col.markdown(f'<div class="metric-box"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                 unsafe_allow_html=True)


def column_config(factors):
    cfg = {GENDER_COL: st.column_config.SelectboxColumn(options=[0, 1], required=True)}
    for f in factors:
        k = kind(f)
        if k == "paps":
            cfg[f] = st.column_config.NumberColumn(f"{f} (PAPS 등급)", min_value=1, max_value=5, step=1,
                                                   help="1등급이 가장 우수 (편성 시 역채점). 비우면 학급 평균")
        elif k == "years":
            cfg[f] = st.column_config.NumberColumn(f, min_value=1, max_value=5, step=1,
                                                   help=", ".join(f"{b}({a})" for a, b in YEARS_LABELS.items()).replace("~", "\\~"))
        elif k == "binary":
            cfg[f] = st.column_config.SelectboxColumn(f, options=[0, 1], help="0=아니오, 1=예")
        elif k == "count5":
            cfg[f] = st.column_config.NumberColumn(f, min_value=1, max_value=5, step=1,
                                                   help="4문항 중 체크 개수(0\\~4)+1 = 1\\~5점")
        else:
            cfg[f] = st.column_config.NumberColumn(f, min_value=1.0, max_value=5.0, step=0.1, format="%.2f",
                                                   help="5점 척도 응답(1\\~5)")
    return cfg


# ─── 페이지 ① AHP 결과 ─────────────────────────────────────────
def table_height(n_rows):
    """머리글 포함 n_rows 행이 스크롤·빈 행 없이 보이는 표 높이(px) — 글씨 17.5px 기준 행 높이 약 38px"""
    return int(37.7 * (n_rows + 1)) + 2


# 순위를 맨 왼쪽 좁은 칸으로, 가중치는 소수점 셋째 자리
RANK_FIRST_CFG = {"순위": st.column_config.NumberColumn("순위", width="small"),
                  "가중치": st.column_config.NumberColumn("가중치", format="%.3f", width="small")}


def html_table(df, center=(), right=(), fmt=None, row_style=None, widths=None, note=None, pad_to=None):
    """AHP 결과용 표 (HTML) — 순위 칸을 좁게·정렬 지정, 행 색 지정, 표 도구 막대 없음
    center/right: 가운데·오른쪽 정렬할 열 · fmt: {열: 형식} · row_style: 행 → CSS 클래스
    widths: {열: CSS 폭} · note: 표 오른쪽 위에 붙여 쓸 글(CR 등) · pad_to: 빈 행으로 채워 맞출 행 수"""
    from html import escape as esc
    fmt, widths = fmt or {}, widths or {}

    def al(c, v=None):
        if c in center:
            return "c"
        if c in right or isinstance(v, (int, float, np.integer, np.floating)):
            return "r"
        return ""
    head = "".join(f'<th class="{al(c) if c in center or c in right else ""}" '
                   f'style="{("width:" + widths[c]) if c in widths else ""}">{esc(str(c))}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cls = row_style(r) if row_style else ""
        tds = "".join(f'<td class="{al(c, r[c])}">{esc(fmt[c].format(r[c]) if c in fmt else str(r[c]))}</td>'
                      for c in df.columns)
        rows.append(f'<tr class="{cls}">{tds}</tr>')
    for _ in range((pad_to or 0) - len(df)):
        rows.append('<tr class="pad">' + "<td>&nbsp;</td>" * len(df.columns) + "</tr>")
    top = f'<div class="cr-label">{esc(note)}</div>' if note else ""
    st.markdown(f'<div>{top}<table class="ahp-tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody>'
                f'</table></div>', unsafe_allow_html=True)


def main_weight_chart(mt):
    """대분류 가중치 가로 막대그래프 — 가중치 순 정렬, 모든 이름표 표시, 막대 끝에 값"""
    import altair as alt
    base = alt.Chart(mt).encode(
        y=alt.Y("대분류 요인:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=220, labelOverlap=False, labelFontSize=14, ticks=False, domain=False)),
        x=alt.X("가중치:Q", title=None, scale=alt.Scale(domain=[0, max(0.5, float(mt["가중치"].max()) + 0.08)]),
                axis=alt.Axis(format=".2f", grid=True, labelFontSize=12)))
    bars = base.mark_bar(color="#2e86de", cornerRadiusEnd=3, size=22)
    text = base.mark_text(align="left", dx=5, fontSize=13, color="#31333f").encode(text=alt.Text("가중치:Q", format=".3f"))
    st.altair_chart((bars + text).properties(height=46 * len(mt)).configure_view(stroke=None),
                    use_container_width=True)


RANK_W = "3.6rem"


def page_ahp():
    st.markdown('<div class="main-title"><span class="mi">bar_chart</span> AHP 분석 결과</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">해당 결과는 2차 델파이(체육교육 전문가 20명)와 AHP(전문가 10명) 분석 결과이다. '
                '종합 가중치는 대분류 가중치 × 하위요인 가중치로 산출하였다.</div>', unsafe_allow_html=True)

    tabs = st.tabs([":material/sports_basketball: 농구", ":material/sports_volleyball: 배구", ":material/sports_baseball: 티볼", ":material/table_chart: 종목 비교"])
    for tab, sport in zip(tabs[:3], SPORTS):
        with tab:
            c1, c2 = st.columns([1, 2], gap="large")
            with c1:
                header("대분류 요인")
                mt = main_table(sport)
                html_table(mt[["순위", "대분류 요인", "가중치"]], center=["순위"], fmt={"가중치": "{:.3f}"},
                           widths={"순위": RANK_W}, note=f"CR = {AHP[sport]['main_cr']:.3f}")
                main_weight_chart(mt)
            with c2:
                header("하위요인 종합 가중치 · 우선순위")
                stt = sub_table(sport).rename(columns={"하위요인 가중치": "하위 가중치", "하위요인 순위": "하위 순위"})
                stt = stt[["대분류", "하위요인", "하위 가중치", "하위 순위", "종합 가중치", "종합 순위"]]
                top = set(top_factors(sport, 5))
                html_table(stt, right=["종합 순위", "하위 순위"],
                           fmt={"하위 가중치": "{:.3f}", "종합 가중치": "{:.3f}"},
                           widths={"종합 순위": "5.2rem", "하위 순위": "5.2rem"},
                           row_style=lambda r: "top" if r["하위요인"] in top
                           else ("excl" if r["하위요인"] in EXCLUDED_FACTORS else ""))
                st.caption("※ 파란색 = 종합 상위 5개 요인(팀 편성 기본값) · 회색 = 리더십(편성 요인에서 제외, 주장 배정으로 대체)")
            with st.expander("대분류별 하위요인 가중치"):
                n_max = max(sum(1 for s in AHP[sport]["sub"] if s[0] == m) for m in MAIN_ORDER)
                for row_names in (MAIN_ORDER[:3], MAIN_ORDER[3:]):
                    cols = st.columns(3, gap="medium")
                    for col, mname in zip(cols, row_names):
                        sdf = pd.DataFrame([(s[3], s[1], s[2]) for s in AHP[sport]["sub"] if s[0] == mname],
                                           columns=["순위", "하위요인", "가중치"]).sort_values("순위")
                        with col:
                            sub_header(mname, mark=False)
                            html_table(sdf, center=["순위"], fmt={"가중치": "{:.3f}"},
                                       widths={"순위": RANK_W, "가중치": "5.5rem"}, note=f"CR = {AHP[sport]['sub_cr'][mname]:.3f}", pad_to=n_max)
            st.caption("※ 가중치는 소수점 셋째 자리 반올림 표기, 순위는 반올림 이전 값 기준.")

    with tabs[3]:
        header("대분류 요인 순위 비교")
        comp = pd.DataFrame({"순위": range(1, 6)})
        for s in SPORTS:
            comp[s] = [f"{r['대분류 요인']} ({r['가중치']:.3f})" for _, r in main_table(s).iterrows()]
        html_table(comp, center=["순위"], widths={"순위": RANK_W})
        header("하위요인 종합 가중치 순위 비교")
        comp2 = pd.DataFrame({"순위": range(1, 19)})
        for s in SPORTS:
            comp2[s] = [f"{r['하위요인']} ({r['종합 가중치']:.3f})" for _, r in sub_table(s).iterrows()]
        html_table(comp2, center=["순위"], widths={"순위": RANK_W})
        header("종목별 종합 가중치 비교")
        chart = pd.DataFrame({s: sub_table(s).set_index("하위요인")["종합 가중치"] for s in SPORTS})
        st.bar_chart(chart.loc[sub_table("농구")["하위요인"]], stack=False, height=420)



# ─── 페이지 ② 팀 편성(교사) ─────────────────────────────────────
def page_teacher():
    st.markdown('<div class="main-title"><span class="mi">balance</span> 체육수업 팀 편성</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title" style="margin-bottom:0.3rem">본 프로그램은 혼합정수선형계획법(MILP)을 이용하여, '
                '선택한 요인마다 각 팀의 평균이 학급 평균에 최대한 가까워지도록 팀을 편성한다.<br>'
                '이때 AHP 가중치가 높은 요인일수록 우선하여 균형을 맞춘다.</div>', unsafe_allow_html=True)
    st.caption("※ 해석상 유의점: 팀별 종합점수를 같게 만드는 방식이 아니므로, 편성 결과의 종합점수에는 "
               "팀 간 차이가 다소 있을 수 있다.")
    ss = st.session_state

    # ① 종목
    header("① 종목")
    sport = st.radio("수업 종목", SPORTS, horizontal=True, key="sport", label_visibility="collapsed")
    stt = sub_table(sport, include_excluded=False)
    ahp_w = dict(zip(stt["하위요인"], stt["종합 가중치"]))

    # ② 팀 편성 요인 — 왼쪽: 요인 선택 · 오른쪽: 적용 가중치 표
    header("② 팀 편성 요인")
    col_sel, col_tbl = st.columns([2, 3], gap="large")
    with col_sel:
        mode = st.radio("요인 선택 방식", ["AHP 상위 요인", "직접 선택"], horizontal=True,
                        label_visibility="collapsed")
        if mode == "AHP 상위 요인":
            top_n = st.number_input("상위 요인 수", 1, len(stt), 5, key="top_n")
            factors = top_factors(sport, int(top_n))
        else:
            options = stt["하위요인"].tolist()
            picked = st.multiselect("편성에 반영할 요인", options, default=top_factors(sport, 5),
                                    format_func=lambda f: f"{f} · {ahp_w[f]:.3f}", key=f"manual_{sport}")
            factors = [f for f in options if f in picked]
        st.caption("적용 가중치는 선택 요인의 AHP 종합 가중치를 합계 100%로 재정규화  \n"
                   "리더십은 편성 요인에서 제외(주장 지정으로 대체)")
    if not factors:
        st.warning("팀 편성 요인을 1개 이상 선택하세요.")
        return
    weights_d = applied_weights(sport, factors)
    with col_tbl:
        wt = stt[stt["하위요인"].isin(factors)][["종합 순위", "대분류", "하위요인"]].copy()
        wt["학생 응답 방식"] = wt["하위요인"].map(lambda f: {
            "paps": "PAPS 등급", "years": "경력 5구간", "binary": "예/아니오",
            "likert": "5점 척도", "count5": "체크리스트(4문항)"}[kind(f)])
        wt["적용 가중치"] = wt["하위요인"].map(lambda f: weights_d[f] * 100)
        st.dataframe(wt, hide_index=True, use_container_width=True, height=table_height(len(wt)) + 12,
                     column_config={
                         "종합 순위": st.column_config.NumberColumn("순위", width="small"),
                         "적용 가중치": st.column_config.ProgressColumn(
                             "적용 가중치", min_value=0, max_value=100, format="%.1f%%")})

    # ③ 학생 데이터
    header("③ 학생 데이터")
    st.caption("※ 이름이 같은 학생이 있으면 이름 뒤에 A, B를 붙여 구분해 입력하세요. (예: 김민준A, 김민준B)")
    src = st.radio("입력 방식", ["학생 참여 (온라인 응답)", "엑셀 업로드", "직접 입력"], horizontal=True,
                   label_visibility="collapsed")
    cols = ["이름", GENDER_COL] + factors
    cfg = column_config(factors)
    sig = f"{sport}_{'|'.join(factors)}"
    df_students = None
    if src.startswith("학생 참여"):
        df_students = student_session_panel(sport, factors, cols, cfg, sig)
    elif src == "엑셀 업로드":
        buf = io.BytesIO()
        pd.DataFrame(columns=cols).to_excel(buf, index=False)
        st.download_button(":material/download: 입력 템플릿 다운로드", buf.getvalue(), file_name=f"팀편성_입력_{sport}.xlsx")
        input_guide("템플릿을 내려받아 열마다 아래 기준으로 입력하세요.")
        up = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
        if up:
            df_up = pd.read_excel(up)
            g = [c for c in df_up.columns if "성별" in str(c)]
            if g:
                df_up = df_up.rename(columns={g[0]: GENDER_COL})
            miss = [c for c in cols if c not in df_up.columns]
            if miss:
                st.warning(f"엑셀에 없는 열: {', '.join(miss)} → 아래 표에서 입력하세요.")
                for c in miss:
                    df_up[c] = None
            df_students = st.data_editor(df_up[cols], use_container_width=True, num_rows="dynamic",
                                         column_config=cfg, key=f"de_up_{sig}")
    else:
        input_guide("표의 각 칸을 아래 기준으로 입력하세요.")
        n_stu = st.number_input("학생 수", min_value=4, max_value=60, value=20)
        base = {"이름": [f"학생{i + 1}" for i in range(n_stu)], GENDER_COL: [0] * n_stu}
        for f in factors:
            base[f] = [0 if kind(f) == "binary" else 3] * n_stu
        df_students = st.data_editor(pd.DataFrame(base), use_container_width=True, num_rows="dynamic",
                                     column_config=cfg, key=f"de_manual_{sig}_{n_stu}")

    names_now = []
    if df_students is not None and len(df_students):
        names_now = [str(x).strip() for x in df_students["이름"].dropna() if str(x).strip()]

    # ④ 팀 설정 · 추가 조건 — 학생 명단을 받은 뒤 설정
    header("④ 팀 설정")
    c1, c2, _ = st.columns([1, 1, 2])
    n_teams = int(c1.number_input("팀 수", 2, 10, 4))
    min_female = int(c2.number_input("팀당 최소 여학생 수", 0, 10, 1))
    with st.expander("추가 조건 (선택사항): 주장 지정 · 같은 팀 결합 · 다른 팀 분리"):
        use_cap = sub_header_toggle("주장을 교사가 미리 지정", key="use_cap")
        st.caption("※ 선택한 주장은 서로 다른 팀에 1명씩 배정되며, 선택한 순서대로 팀 A, 팀 B …가 됩니다.")
        captains = []
        if use_cap:
            captains = st.multiselect(f"주장 선택 (최대 {n_teams}명, 선택 순서 = 팀 번호)", names_now,
                                      max_selections=n_teams, key="captains")
            if captains and len(captains) < n_teams:
                st.caption(f"주장이 {len(captains)}명이므로 팀 {team_label(len(captains) + 1)}\\~{team_label(n_teams)}에는 주장이 없습니다.")
        sub_header("같은 팀 결합 · 다른 팀 분리")
        st.caption("※ 짝은 한 줄에 두 명씩 지정하며, 줄을 추가해 여러 쌍을 지정할 수 있습니다.")
        empty_pairs = pd.DataFrame({"학생 A": pd.Series(dtype="object"), "학생 B": pd.Series(dtype="object")})
        pair_cfg = {c: st.column_config.SelectboxColumn(c, options=names_now) for c in ("학생 A", "학생 B")}
        c1, c2 = st.columns(2)
        with c1:
            sub_header("같은 팀으로 결합할 짝", mark=False)
            same_df = st.data_editor(empty_pairs, num_rows="dynamic", use_container_width=True,
                                     column_config=pair_cfg, hide_index=True, key="same_pairs")
        with c2:
            sub_header("다른 팀으로 분리할 짝", mark=False)
            diff_df = st.data_editor(empty_pairs, num_rows="dynamic", use_container_width=True,
                                     column_config=pair_cfg, hide_index=True, key="diff_pairs")
    same_pairs, diff_pairs = clean_pairs(same_df), clean_pairs(diff_df)

    # ⑤ 실행
    header("⑤ 팀 편성 실행")
    if st.button(":material/rocket_launch: 팀 편성 실행", type="primary", use_container_width=True):
        ss.pop("result", None)
        if df_students is None or len(df_students) == 0:
            st.error("학생 데이터를 입력하세요.")
            return
        df = df_students.copy()
        df["이름"] = df["이름"].astype(str).str.strip()
        df = df[~df["이름"].str.lower().isin(["", "nan", "none"])]
        df = df.drop_duplicates("이름", keep="last").reset_index(drop=True)
        n = len(df)
        genders = [parse_gender(v) for v in df[GENDER_COL]]
        names = df["이름"].tolist()
        errs = []
        if n < n_teams * 2:
            errs.append(f"학생 수({n}명)가 너무 적습니다. 팀 수 × 2명 이상 필요합니다.")
        if sum(genders) < min_female * n_teams:
            errs.append(f"여학생 {sum(genders)}명 < {n_teams}팀 × {min_female}명. '팀당 최소 여학생 수'를 줄이세요.")
        errs += check_pairs(names, n_teams, same_pairs, diff_pairs, captains)
        if errs:
            for e in errs:
                st.error(e, icon=":material/error:")
            return
        ok, ver = pulp_compatible()
        if not ok:
            st.error(f"설치된 PuLP {ver} 버전은 이 프로그램과 호환되지 않습니다 (PuLP 3.x 이상은 사용법이 바뀜).\n\n"
                     f"터미널에서 앱을 끄고(Ctrl+C) 아래 명령으로 PuLP 2.9를 설치한 뒤 다시 실행하세요.")
            st.code(PULP_FIX_CMD, language="powershell")
            return
        raw, score, z, warns = build_score_matrix(df, factors)
        for w in warns:
            st.warning(w, icon=":material/warning:")
        with st.spinner("최적 팀 편성 계산 중..."):
            res = run_milp(names, genders, z[factors].values, [weights_d[f] for f in factors],
                           n_teams, min_female, same_pairs, diff_pairs, captains)
        if res.get("error") == "timeout":
            st.error("제한시간(60초) 안에 최적해를 확정하지 못했습니다. "
                     "팀 편성 요인 수나 팀 수를 줄이거나, 묶기·분리·주장 조건을 완화한 뒤 다시 실행해 주세요.")
            return
        if res.get("error"):
            st.error("모든 제약조건을 동시에 만족하는 편성이 없습니다. "
                     "제약 조건(여학생 수·묶기·분리·주장)을 확인해 주세요.")
            return
        stu, team = summarize(names, genders, raw, score, weights_d, res["assign"], n_teams, captains)
        ss["result"] = {"stu": stu, "team": team, "objective": res["objective"], "sport": sport,
                        "factors": factors, "weights": weights_d, "n_teams": n_teams,
                        "min_female": min_female, "mode": mode,
                        "assign": res["assign"], "captains": list(captains),
                        "code": ss.get("session_code") if src.startswith("학생 참여") else None,
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "same_pairs": same_pairs, "diff_pairs": diff_pairs}
        st.rerun()
    if "result" in ss:
        show_result(ss["result"])


def make_qr_png(text):
    """학생 참여 링크 → QR 코드 PNG(bytes). qrcode 라이브러리가 없으면 None"""
    try:
        import qrcode
    except ImportError:
        return None
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=2)
    qr.add_data(text)
    qr.make(fit=True)
    buf = io.BytesIO()
    qr.make_image(fill_color="black", back_color="white").save(buf)
    return buf.getvalue()


def student_session_panel(sport, factors, cols, cfg, sig):
    ss = st.session_state
    with st.container(border=True):
        tab_new, tab_load = st.tabs(["새 참여 세션 만들기", "기존 참여코드 불러오기"])
        with tab_new:
            sub_header("세션 이름")
            title = st.text_input("세션 이름", value="", placeholder=f"예: 2학년 3반 {sport} 수업",
                                  label_visibility="collapsed")
            all_f = sub_table(sport, include_excluded=False)["하위요인"].tolist()
            ask_all = sub_header_toggle("모든 요인을 설문받기", key=f"ask_all_{sport}")
            st.caption("※ 리더십을 제외한 모든 요인을 학생에게 설문하여, 응답을 받은 뒤 팀 편성 요인을 바꿔도 "
                       "다시 설문할 필요가 없습니다.")
            stu_f = all_f if ask_all else [f for f in all_f if f in factors]
            sub_header("설문 문항 미리보기")
            with st.expander(f"{len(stu_f)}개 요인 문항 보기"):
                st.dataframe(questionnaire_table(sport, stu_f), hide_index=True, use_container_width=True)
            st.caption("※ 참여코드와 학생 응답은 생성 후 6개월이 지나면 자동으로 삭제됩니다.")
            if st.button("참여코드 생성", type="primary"):
                ss["session_code"] = create_session(title.strip() or f"{datetime.now():%m/%d} {sport} 수업",
                                                    sport, stu_f)
                ss.pop("result", None)
        with tab_load:
            c = st.text_input("참여코드 입력", key="load_code")
            if st.button("불러오기"):
                if get_session(c):
                    ss["session_code"] = c.strip().upper()
                    ss.pop("result", None)
                    ss.pop("restore_code", None)
                else:
                    ss["restore_code"] = c.strip().upper()
            rc = ss.get("restore_code")
            if rc and rc == c.strip().upper():
                st.warning(f"서버에 참여코드 {rc}의 자료가 없습니다. 저장해 둔 학생 데이터 파일을 올리면 복원됩니다.",
                           icon=":material/warning:")
                up_b = st.file_uploader("학생 데이터 파일 (.xlsx)", type=["xlsx"], key=f"restore_{rc}")
                if up_b is not None:
                    ok, msg = restore_session_backup(up_b, rc)
                    if ok:
                        ss["session_code"] = rc
                        ss.pop("result", None)
                        ss.pop("restore_code", None)
                        st.rerun()
                    else:
                        st.error(msg)

    code = ss.get("session_code")
    if not code:
        st.info("참여코드를 만들면 학생들이 각자 휴대폰·태블릿으로 설문에 응답합니다.")
        return None
    sess = get_session(code)
    if not sess:
        st.error("세션 정보를 찾을 수 없습니다.")
        return None

    try:
        base = st.context.url.split("?")[0]
    except Exception:
        base = "<앱 주소>"
    link = f"{base}?code={code}"
    qr_png = make_qr_png(link) if base.startswith("http") else None

    # 참여코드 · QR 코드
    c1, c2, _ = st.columns([1, 1, 2], vertical_alignment="center")
    c1.markdown(f'<div class="code-box">{code}</div>', unsafe_allow_html=True)
    with c2:
        if qr_png:
            st.image(qr_png, caption="휴대폰 카메라로 스캔", width=160)
        else:
            st.caption("QR 코드를 만들 수 없습니다. (qrcode 라이브러리 설치 필요: pip install qrcode)")
    if qr_png:
        with st.expander(":material/zoom_in: QR 코드 크게 보기 (교실 화면 띄우기용)"):
            st.image(qr_png, width=420)
            st.markdown(f'<div class="code-box">{code}</div>', unsafe_allow_html=True)
            st.download_button(":material/download: QR 코드 이미지 저장", qr_png,
                               file_name=f"참여코드_{code}_QR.png", mime="image/png")
    if sess["sport"] != sport:
        st.warning(f"이 세션은 '{sess['sport']}'로 만들어졌습니다. 현재 선택 종목은 '{sport}'입니다.")

    # 응답 새로고침 · 응답 마감 · 상태 · 제출 인원
    subs = load_submissions(code, factors)
    n_f = int((subs[GENDER_COL] == 1).sum()) if len(subs) else 0
    b1, b2, b3 = st.columns([1, 1, 2], vertical_alignment="center")
    if b1.button(":material/refresh: 응답 새로고침", use_container_width=True):
        st.rerun()
    if b2.button(":material/block: 응답 마감" if sess["is_open"] else ":material/lock_open: 응답 다시 열기",
                 use_container_width=True):
        set_session_open(code, not sess["is_open"])
        st.rerun()
    b3.markdown(f"상태: **{'응답 받는 중' if sess['is_open'] else '마감'}** &nbsp;·&nbsp; "
                f"**제출 {len(subs)}명** · 여학생 {n_f}명")

    # 설문하지 않은 요인을 편성에 쓰는 경우 즉시 경고
    not_asked = [f for f in factors if f not in sess["factors"]]
    if not_asked and len(subs):
        st.warning(f"'{', '.join(not_asked)}'은(는) 이 참여코드에서 설문하지 않은 요인이라 모든 학생의 값이 비어 있습니다. "
                   "아래 응답 관리 표에 직접 입력하지 않으면 편성에 반영되지 않습니다.", icon=":material/warning:")

    # 응답 관리 (학생 데이터 확인·수정) — 기본은 접힘
    edited = None
    with st.expander("응답 관리 (학생 데이터 확인·수정)", expanded=False):
        if len(subs) == 0:
            st.caption("아직 제출한 학생이 없습니다. 응답이 들어오면 '응답 새로고침'을 누르세요.")
        else:
            note, bk = st.columns([3, 1], vertical_alignment="center")
            note.caption("※ 잘못 제출된 학생은 표 왼쪽 끝 칸을 눌러 행을 선택한 뒤, 표 위의 휴지통 아이콘(또는 Delete 키)으로 삭제하세요.")
            bk.download_button(":material/save: 학생 데이터 저장",
                               export_session_backup(code),
                               file_name=f"학생데이터_{code}.xlsx", use_container_width=True, key="save_student",
                               help="학생 데이터(교사 수정값 포함)를 선생님 PC에 저장합니다. "
                                    "서버 자료가 지워져도 '기존 참여코드 불러오기'에서 이 파일로 복원할 수 있습니다.")
            ver = hashlib.md5(subs[cols].to_json(force_ascii=False).encode("utf-8")).hexdigest()[:10]
            edited = st.data_editor(subs[cols], use_container_width=True, num_rows="dynamic",
                                    column_config=cfg, key=f"de_sess_{code}_{sig}_{ver}")
            if sync_teacher_table(code, subs, edited, factors):
                st.rerun()
    return edited


def result_tables(r):
    """편성 결과 → (학생별 결과, 팀별 요약, 편성 설정) 표"""
    stu, team, factors, n_teams = r["stu"], r["team"], r["factors"], r["n_teams"]
    sc = team["종합 점수 평균"].tolist()
    stu_view = stu.assign(**{"배정 팀": stu["배정 팀"].map(lambda t: f"팀 {team_label(t)}")})
    setting = pd.DataFrame([
        ("종목", r["sport"]), ("요인 선택 방식", r["mode"]),
        ("가중치 방식", "AHP 종합 가중치 재정규화"),
        ("점수 변환", "PAPS 역채점 → 요인별 학급 표준편차로 나눔"),
        ("팀 수", n_teams), ("팀당 최소 여학생", r["min_female"]),
        ("주장 사전 배정", ", ".join(r["captains"]) or "없음"),
        ("결합(같은 팀) 짝", " / ".join(f"{a}–{b}" for a, b in r["same_pairs"]) or "없음"),
        ("분리(다른 팀) 짝", " / ".join(f"{a}–{b}" for a, b in r["diff_pairs"]) or "없음"),
        ("목적함수 최솟값", round(r["objective"], 6)),
        ("팀 종합점수 최고·최저·차·표준편차", f"{max(sc):.1f} · {min(sc):.1f} · {max(sc) - min(sc):.1f} · "
                                       f"{float(np.std(sc, ddof=0)):.2f}"),
        ("편성 일시", r.get("time", "")),
    ] + [(f"가중치: {f}", round(r["weights"][f], 4)) for f in factors], columns=["항목", "값"])
    return stu_view, team, setting


def write_result_sheets(xw, r):
    stu_view, _, _ = result_tables(r)
    stu_view.to_excel(xw, sheet_name="편성 결과", index=False)


def show_result(r):
    ss = st.session_state
    stu, team, factors, n_teams = r["stu"], r["team"], r["factors"], r["n_teams"]
    st.success("팀 편성 완료!", icon=":material/check_circle:")
    out = io.BytesIO()
    with pd.ExcelWriter(out) as xw:
        write_result_sheets(xw, r)                      # 1번 시트: 편성 결과
        if r.get("code"):                               # 학생 참여 방식: 학생 데이터 · 복원용 시트를 뒤에
            write_student_sheets(xw, r["code"])
            write_backup_sheets(xw, r["code"])
    fname = f"팀편성결과_{r['code']}.xlsx" if r.get("code") else f"팀편성결과_{r['sport']}.xlsx"
    h1, h2 = st.columns([3, 1], vertical_alignment="bottom")
    with h1:
        header("⑥ 편성 결과")
    h2.download_button(":material/save: 팀 편성 결과 데이터 저장", out.getvalue(), file_name=fname,
                       key="save_result", use_container_width=True)
    sc = team["종합 점수 평균"].tolist()

    # 1. 팀별 명단 · 팀별 종합점수
    sub_header("팀별 명단 · 팀별 종합점수(0~100점)")
    gmap = dict(zip(stu["이름"], stu[GENDER_COL]))
    caps = set(r["captains"])
    for start in range(1, n_teams + 1, 2):
        tl = list(range(start, min(start + 2, n_teams + 1)))
        for col, t in zip(st.columns(len(tl)), tl):
            mem = stu[stu["배정 팀"] == t]["이름"].tolist()
            badges = " ".join(
                f'<span class="badge{" badge-f" if gmap[m] == 1 else ""}{" badge-c" if m in caps else ""}">'
                f'{"★ " if m in caps else ""}{m}</span>' for m in mem)
            s = team.loc[t - 1, "종합 점수 평균"]
            col.markdown(f'<div class="team-card"><h4><span class="mi">groups</span> 팀 {team_label(t)} &nbsp;<small style="color:#888">'
                         f'팀원 수 {len(mem)}명 · 종합점수 {s:.1f}점</small></h4>{badges}</div>', unsafe_allow_html=True)

    # 2. 팀 간 균형 지표
    sub_header("팀 간 균형 지표")
    m1, m2, m3, m4, m5 = st.columns(5)
    metric(m1, f"{min(sc):.1f}", "최저 팀 종합점수")
    metric(m2, f"{max(sc):.1f}", "최고 팀 종합점수")
    metric(m3, f"{max(sc) - min(sc):.1f}", "팀 간 종합점수 차")
    metric(m4, f"{float(np.std(sc, ddof=0)):.2f}", "팀 종합점수 표준편차")
    metric(m5, f"{r['objective']:.4f}", "목적함수 최솟값")
    st.caption("※ 목적함수 최솟값은 요인별로 팀 평균과 학급 평균의 차이(표준편차 단위)에 가중치를 곱해 모두 더한 값의 최솟값으로, "
               "0에 가까울수록 팀 간 균형이 좋음을 의미한다.")

    # 3. 팀별 요인 평균
    sub_header("팀별 요인 평균(1~5점; 체육특기자 여부는 인원 수)")
    if any(kind(f) == "paps" for f in factors):
        st.caption("※ PAPS 요인은 등급 평균으로, 값이 낮을수록 우수함을 의미한다.")
    st.dataframe(team, hide_index=True, use_container_width=True)
    stu_view, _, _ = result_tables(r)
    with st.expander(":material/list: 학생별 결과"):
        st.dataframe(stu_view, hide_index=True, use_container_width=True)


    code = ss.get("session_code")
    if code:
        c1, c2 = st.columns(2)
        if c1.button(":material/campaign: 학생에게 결과 공개", use_container_width=True):
            publish_results(code, {"assign": {k: int(v) for k, v in r["assign"].items()},
                                   "captains": r["captains"]})
            ss["pub_msg"] = f"공개 완료 — 학생들이 참여코드 {code}로 자기 팀을 확인할 수 있습니다."
            st.rerun()
        if c2.button(":material/visibility_off: 결과 공개 취소", use_container_width=True):
            publish_results(code, None)
            ss["pub_msg"] = "공개를 취소했습니다."
            st.rerun()
        if ss.get("pub_msg"):
            st.success(ss.pop("pub_msg"))


# ─── 페이지 ③ 학생 참여 ─────────────────────────────────────────
def page_student(prefill=""):
    st.markdown('<div class="main-title"><span class="mi">front_hand</span> 팀 편성 설문</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">정답이 있는 시험이 아닙니다. 솔직하게 응답할수록 균형 있는 팀이 만들어집니다.</div>',
                unsafe_allow_html=True)
    code = st.text_input("참여코드", value=prefill, max_chars=6).strip().upper()
    if not code:
        return
    sess = get_session(code)
    if not sess:
        st.error("참여코드를 다시 확인해 주세요.")
        return
    sport = sess["sport"]
    st.markdown(f"### {sess['title']} · {sport}")

    t_form, t_result = st.tabs([":material/edit_note: 설문 응답", ":material/visibility: 내 팀 확인"])
    with t_form:
        if not sess["is_open"]:
            st.warning("응답이 마감되었습니다.")
        else:
            with st.form("stu_form"):
                name = st.text_input("이름")
                st.caption("※ 우리 반에 이름이 같은 친구가 있으면 이름 뒤에 A, B를 붙여 입력하세요. (예: 김민준A, 김민준B)")
                gender = st.radio("성별", ["남", "여"], horizontal=True, index=None)
                answers = {}
                for gname, gf in GROUPS:
                    fs = [f for f in gf if f in sess["factors"]]
                    if not fs:
                        continue
                    st.divider()
                    st.markdown(f"#### {gname}")
                    if any(kind(f) == "count5" for f in fs):
                        st.caption("내가 알고 있거나 할 수 있는 것에 모두 체크하세요. (해당 없으면 체크하지 않아도 됩니다)")
                    if any(kind(f) == "likert" for f in fs):
                        st.caption("1 전혀 그렇지 않다 · 2 그렇지 않다 · 3 보통이다 · 4 그렇다 · 5 매우 그렇다")
                    for f in fs:
                        for iid, q in items(sport, f):
                            k = kind(f)
                            if k == "count5":
                                answers[iid] = int(st.checkbox(q, key=iid))
                                continue
                            st.markdown(f'<div class="q">{q}</div>', unsafe_allow_html=True)
                            if k == "paps":
                                a = st.radio(q, PAPS_OPTIONS, horizontal=True, index=None, key=iid,
                                             label_visibility="collapsed")
                                answers[iid] = int(a[0]) if a else None
                            elif k == "years":
                                a = st.radio(q, list(YEARS_LABELS), horizontal=True, index=None, key=iid,
                                             format_func=lambda v: YEARS_LABELS[v], label_visibility="collapsed")
                                answers[iid] = a
                            elif k == "binary":
                                a = st.radio(q, ["아니오", "예"], horizontal=True, index=None, key=iid,
                                             label_visibility="collapsed")
                                answers[iid] = None if a is None else (1 if a == "예" else 0)
                            else:
                                answers[iid] = st.radio(q, [1, 2, 3, 4, 5], horizontal=True, index=None,
                                                        key=iid, label_visibility="collapsed")
                ok = st.form_submit_button("제출하기", type="primary", use_container_width=True)
            if ok:
                blank = [iid for iid, v in answers.items() if v is None]
                if not name.strip():
                    st.error("이름을 입력하세요.")
                elif gender is None:
                    st.error("성별을 선택하세요.")
                elif blank:
                    st.error(f"응답하지 않은 문항이 {len(blank)}개 있습니다.")
                elif not get_session(code)["is_open"]:
                    st.error("응답이 마감되었습니다.")
                else:
                    fv = {}
                    for f in sess["factors"]:
                        vals = [answers[iid] for iid, _ in items(sport, f)]
                        if kind(f) == "count5":
                            fv[f] = COUNT_TO_SCORE[int(sum(vals))]
                            continue
                        nums = [v for v in vals if isinstance(v, (int, float))]
                        fv[f] = round(float(np.mean(nums)), 2) if nums else None
                    save_submission(code, name.strip(), 1 if gender == "여" else 0, answers, fv)
                    st.success(f"{name.strip()} 학생, 제출되었습니다! (같은 이름으로 다시 제출하면 수정됩니다)")
                    st.balloons()
    with t_result:
        res = sess["results"]
        if not res:
            st.info("아직 팀 편성 결과가 공개되지 않았습니다.")
        else:
            assign, caps = res.get("assign", {}), set(res.get("captains", []))
            me = st.text_input("내 이름", key="me").strip()
            if me:
                if me not in assign:
                    st.error("편성 결과에서 이름을 찾을 수 없습니다.")
                else:
                    t = assign[me]
                    mates = [n for n, v in assign.items() if v == t]
                    st.success(f"**{me}** 학생은 **팀 {team_label(t)}** 입니다!" + (" (주장 ★)" if me in caps else ""))
                    st.markdown("팀원: " + " ".join(
                        f'<span class="badge">{"★ " if m in caps else ""}{m}</span>' for m in mates),
                        unsafe_allow_html=True)


# ─── 메인 ───────────────────────────────────────────────────────
TEACHER_GUIDE = """
**:material/checklist: 교사 사용 방법**

1. **② 팀 편성(교사)** 에서 종목 선택
2. 팀 편성 요인 선택
3. 참여코드 / QR 코드 생성
4. 응답자 확인 → 응답 마감
5. 응답 관리 (학생 데이터 확인·수정)
6. 팀 수 · 팀당 최소 여학생 수 입력
7. 주장 · 결합 · 분리 지정 (선택사항)
8. :material/rocket_launch: 팀 편성 실행

<div style="color:#e03131; font-size:0.85rem; line-height:1.5; margin-top:-0.6rem;">※ Streamlit 서버 특성상 저장된 자료가<br>초기화될 수 있으므로, 필요한 경우 반드시<br>학생/팀 편성 결과 데이터를 저장하시오.</div>
"""


def main():
    st.set_page_config(page_title="체육수업 팀 편성 프로그램", page_icon="🏀", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if not st.session_state.get("_purged"):      # 접속할 때마다 한 번, 보관 기간이 지난 자료 정리
        try:
            purge_old_sessions()
        except Exception:
            pass
        st.session_state["_purged"] = True
    qcode = st.query_params.get("code", "")
    if qcode:   # 학생용 링크 → 학생 화면만
        page_student(qcode)
        return
    with st.sidebar:
        st.markdown('<div class="side-title"><span class="mi">sports_basketball</span> 농구 · '
                    '<span class="mi">sports_volleyball</span> 배구 · <span class="mi">sports_baseball</span> 티볼'
                    '<br>팀 편성 프로그램</div>', unsafe_allow_html=True)
        page = st.radio("메뉴", ["① AHP 결과", "② 팀 편성(교사)", "③ 학생 참여"], label_visibility="collapsed")
        st.divider()
        st.markdown(TEACHER_GUIDE, unsafe_allow_html=True)
    pw = os.environ.get("TEACHER_PASSWORD", "")
    if page == "② 팀 편성(교사)" and pw and not st.session_state.get("auth"):
        if st.text_input("교사 비밀번호", type="password") == pw:
            st.session_state["auth"] = True
            st.rerun()
        return
    if page == "① AHP 결과":
        page_ahp()
    elif page == "② 팀 편성(교사)":
        page_teacher()
    else:
        page_student()


def _running_in_streamlit():
    """streamlit run 으로 실행 중인지 확인"""
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            return get_script_run_ctx() is not None
        except Exception:
            return True


if __name__ == "__main__":
    if _running_in_streamlit():
        main()
    else:
        # VS Code ▷ 버튼(python 파일.py)으로 실행한 경우 → streamlit run 으로 자동 재실행
        import subprocess
        import sys
        print("Streamlit 앱을 실행합니다. 브라우저가 열리지 않으면 http://localhost:8501 에 접속하세요.")
        print("종료하려면 이 터미널에서 Ctrl+C 를 누르세요.")
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
