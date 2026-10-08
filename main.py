#!/usr/bin/env python3
"""Лабораторная работа № 1: генетический алгоритм для функции Швефеля.

Программа использует только стандартную библиотеку Python.
Запуск одного эксперимента: python3 main.py --experiment
"""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


LOWER_BOUND = -500.0
UPPER_BOUND = 500.0
DIMENSION = 5  # d = 5 + (6 mod 6)


def schwefel(point):
    """Возвращает значение функции Швефеля, которое нужно минимизировать."""
    total = 0.0
    for value in point:
        total += value * math.sin(math.sqrt(abs(value)))
    return 418.9829 * len(point) - total


def make_individual(generator):
    return [generator.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(DIMENSION)]


def clip(value):
    """Не выпускает координату за границы области поиска."""
    return max(LOWER_BOUND, min(UPPER_BOUND, value))


def tournament(population, generator, tournament_size):
    """Выбирает лучшего из нескольких случайных кандидатов."""
    candidates = [generator.choice(population) for _ in range(tournament_size)]
    return min(candidates, key=lambda item: item[1])


def make_child(parent_a, parent_b, config, generator):
    """Арифметическое скрещивание и простая гауссовская мутация."""
    first, second = parent_a[0], parent_b[0]
    if generator.random() < config["crossover_probability"]:
        weight = generator.random()
        child = [weight * a + (1.0 - weight) * b for a, b in zip(first, second)]
    else:
        child = first.copy()

    for index in range(DIMENSION):
        if generator.random() < config["mutation_probability"]:
            child[index] += generator.gauss(0.0, config["mutation_sigma"])
        child[index] = clip(child[index])
    return child


def run_ga(config, seed):
    """Выполняет один запуск ГА и возвращает результат и траекторию лучшего."""
    generator = random.Random(seed)
    population = []
    for _ in range(config["population_size"]):
        individual = make_individual(generator)
        population.append((individual, schwefel(individual)))

    history = [min(population, key=lambda item: item[1])[1]]
    evaluations = config["population_size"]

    # Элитная особь переносится в следующее поколение без изменений.
    for _ in range(1, config["generations"]):
        population.sort(key=lambda item: item[1])
        next_population = [population[0]]
        while len(next_population) < config["population_size"]:
            parent_a = tournament(population, generator, config["tournament_size"])
            parent_b = tournament(population, generator, config["tournament_size"])
            child = make_child(parent_a, parent_b, config, generator)
            next_population.append((child, schwefel(child)))
            evaluations += 1
        population = next_population
        history.append(min(population, key=lambda item: item[1])[1])

    best = min(population, key=lambda item: item[1])
    return {
        "best_value": best[1],
        "best_point": best[0],
        "evaluations": evaluations,
        "history": history,
    }


def run_random_search(evaluations, seed):
    """Случайная выборка с тем же числом вычислений функции, что у ГА."""
    generator = random.Random(seed)
    best_value = float("inf")
    best_point = None
    for _ in range(evaluations):
        point = make_individual(generator)
        value = schwefel(point)
        if value < best_value:
            best_value = value
            best_point = point
    return {"best_value": best_value, "best_point": best_point, "evaluations": evaluations}


def mean(values):
    return sum(values) / len(values)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def create_convergence_chart(path, trajectories):
    """Рисует SVG без внешних библиотек: минимум, среднее и максимум серии."""
    width, height = 900, 500
    left, right, top, bottom = 75, 30, 35, 65
    generations = len(trajectories[0])
    lower = min(min(line) for line in trajectories)
    upper = max(max(line) for line in trajectories)
    margin = (upper - lower) * 0.05 or 1.0
    # Для функции Швефеля значения на данной области неотрицательны.
    lower = max(0.0, lower - margin)
    upper += margin

    def x(value):
        return left + value * (width - left - right) / (generations - 1)

    def y(value):
        return top + (upper - value) * (height - top - bottom) / (upper - lower)

    minimum = [min(row[g] for row in trajectories) for g in range(generations)]
    average = [mean([row[g] for row in trajectories]) for g in range(generations)]
    maximum = [max(row[g] for row in trajectories) for g in range(generations)]

    def polyline(values, color):
        points = " ".join(f"{x(i):.1f},{y(value):.1f}" for i, value in enumerate(values))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:14px}.axis{stroke:#333}.grid{stroke:#ddd}</style>',
        f'<text x="{width / 2}" y="22" text-anchor="middle">Сходимость ГА: 20 запусков, базовая конфигурация</text>',
    ]
    for fraction in range(6):
        value = lower + (upper - lower) * fraction / 5
        yy = y(value)
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text x="{left-8}" y="{yy+5:.1f}" text-anchor="end">{value:.1f}</text>')
    for generation in range(0, generations, 20):
        xx = x(generation)
        parts.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{height-bottom}"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" text-anchor="middle">{generation + 1}</text>')
    parts.extend([
        f'<line class="axis" x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}"/>',
        polyline(maximum, "#e15759"), polyline(average, "#4e79a7"), polyline(minimum, "#59a14f"),
        f'<text x="{width / 2}" y="{height-12}" text-anchor="middle">Поколение</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle">Лучшее значение f(x)</text>',
        f'<rect x="{width-260}" y="40" width="210" height="66" fill="white" stroke="#aaa"/>',
        f'<line x1="{width-245}" x2="{width-220}" y1="58" y2="58" stroke="#59a14f" stroke-width="3"/><text x="{width-210}" y="63">минимум</text>',
        f'<line x1="{width-245}" x2="{width-220}" y1="78" y2="78" stroke="#4e79a7" stroke-width="3"/><text x="{width-210}" y="83">среднее</text>',
        f'<line x1="{width-245}" x2="{width-220}" y1="98" y2="98" stroke="#e15759" stroke-width="3"/><text x="{width-210}" y="103">максимум</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def create_comparison_chart(path, summary_rows):
    """Столбчатая диаграмма среднего лучшего результата трёх методов."""
    width, height = 900, 500
    left, right, top, bottom = 90, 30, 35, 85
    values = [row["mean"] for row in summary_rows]
    upper = max(values) * 1.1
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    labels = ["ГА, p=0.15", "ГА, p=0.40", "Случайный поиск"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:14px}.axis{stroke:#333}.grid{stroke:#ddd}</style>',
        f'<text x="{width/2}" y="22" text-anchor="middle">Средний лучший результат (20 запусков)</text>',
    ]
    for fraction in range(6):
        value = upper * fraction / 5
        yy = top + (upper - value) * (height-top-bottom) / upper
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text x="{left-8}" y="{yy+5:.1f}" text-anchor="end">{value:.1f}</text>')
    bar_width = 140
    centers = [250, 450, 650]
    for index, (center, value) in enumerate(zip(centers, values)):
        bar_height = value * (height-top-bottom) / upper
        yy = height - bottom - bar_height
        parts.append(f'<rect x="{center-bar_width/2}" y="{yy:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[index]}"/>')
        parts.append(f'<text x="{center}" y="{yy-8:.1f}" text-anchor="middle">{value:.2f}</text>')
        parts.append(f'<text x="{center}" y="{height-bottom+24}" text-anchor="middle">{labels[index]}</text>')
    parts.extend([
        f'<line class="axis" x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}"/>',
        f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle">Среднее значение f(x), меньше — лучше</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def experiment(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_directory"])
    output.mkdir(exist_ok=True)
    run_count = config["run_count"]
    base_seed = config["base_seed"]

    groups = [
        ("ga_base", config["base_ga"]),
        ("ga_high_mutation", config["high_mutation_ga"]),
    ]
    runs = []
    trajectories = []
    all_values = {}
    sample_result = None
    for name, ga_config in groups:
        values = []
        for number in range(run_count):
            seed = base_seed + number
            result = run_ga(ga_config, seed)
            values.append(result["best_value"])
            runs.append([name, number + 1, seed, result["evaluations"], result["best_value"], *result["best_point"]])
            if name == "ga_base":
                trajectories.append(result["history"])
                if number == 0:
                    sample_result = result
        all_values[name] = values

    random_values = []
    evaluation_budget = sample_result["evaluations"]
    for number in range(run_count):
        seed = base_seed + number
        result = run_random_search(evaluation_budget, seed)
        random_values.append(result["best_value"])
        runs.append(["random_search", number + 1, seed, result["evaluations"], result["best_value"], *result["best_point"]])
    all_values["random_search"] = random_values

    write_csv(output / "runs.csv", ["method", "run", "seed", "evaluations", "best_value", "x1", "x2", "x3", "x4", "x5"], runs)
    trajectory_rows = []
    for generation in range(len(trajectories[0])):
        values = [line[generation] for line in trajectories]
        trajectory_rows.append([generation + 1, min(values), mean(values), max(values)])
    write_csv(output / "trajectory.csv", ["generation", "minimum", "mean", "maximum"], trajectory_rows)

    names = {"ga_base": "ГА: базовая мутация 0.15", "ga_high_mutation": "ГА: мутация 0.40", "random_search": "Случайный поиск"}
    summary_rows = []
    for key in ("ga_base", "ga_high_mutation", "random_search"):
        values = all_values[key]
        summary_rows.append({
            "method": names[key], "best": min(values), "mean": mean(values),
            "median": statistics.median(values), "std": statistics.stdev(values), "worst": max(values),
        })
    write_csv(output / "summary.csv", ["method", "best", "mean", "median", "std", "worst"],
              [[row["method"], row["best"], row["mean"], row["median"], row["std"], row["worst"]] for row in summary_rows])
    create_convergence_chart(output / "convergence.svg", trajectories)
    create_comparison_chart(output / "comparison.svg", summary_rows)

    print(f"Готово. Файлы эксперимента: {output}")
    print(f"Бюджет одного запуска: {evaluation_budget} вычислений функции")
    for row in summary_rows:
        print(f'{row["method"]}: среднее = {row["mean"]:.4f}, лучшее = {row["best"]:.4f}')


def demo(config_path, seed):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run_ga(config["base_ga"], seed)
    print(f"seed = {seed}")
    print(f"Лучшее значение f(x) = {result['best_value']:.6f}")
    print("Точка x =", ", ".join(f"{value:.6f}" for value in result["best_point"]))
    print(f"Вычислений функции = {result['evaluations']}")


def main():
    parser = argparse.ArgumentParser(description="ГА для функции Швефеля, вариант 6")
    parser.add_argument("--config", default="config.json", help="путь к файлу настроек")
    parser.add_argument("--seed", type=int, default=20261001, help="seed для одиночного запуска")
    parser.add_argument("--experiment", action="store_true", help="провести серию экспериментов")
    args = parser.parse_args()
    config_path = Path(args.config)
    if args.experiment:
        experiment(config_path)
    else:
        demo(config_path, args.seed)


if __name__ == "__main__":
    main()
