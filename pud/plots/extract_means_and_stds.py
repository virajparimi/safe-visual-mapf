import numpy as np
from pathlib import Path


def extract_single_agent_metrics(records, search_based=(False, None)):
    success_rate = 0.0
    steps = []
    rewards = []
    cumulative_costs = []
    for idx, record in enumerate(records):
        if len(record) == 0:
            if search_based[0] and search_based[1][idx]["success"]:
                success_rate += 1
                steps.append(search_based[1][idx]["steps"])
                rewards.append(search_based[1][idx]["rewards"])
                cumulative_costs.append(search_based[1][idx]["cumulative_costs"])
            continue
        elif record["success"]:
            success_rate += 1
            steps.append(record["steps"])
            rewards.append(record["rewards"])
            cumulative_costs.append(record["cumulative_costs"])

    metrics = {
        "steps": steps,
        "rewards": rewards,
        "cumulative_costs": cumulative_costs,
        "success_rate": success_rate / len(records),
    }
    return metrics


def extract_multi_agent_metrics(records, num_agents, search_based=(False, None)):
    success_rate = 0.0
    mean_steps = []
    mean_rewards = []
    mean_cumulative_costs = []
    if search_based[0]:
        fallback_num = 0
        fallback_successes = 0
    for idx, record in enumerate(records):
        steps = []
        rewards = []
        successes = []
        cumulative_costs = []
        if len(record[0]) == 0:
            if search_based[0]:
                fallback_num += 1
                for i in range(num_agents):
                    successes.append(search_based[1][idx][i]["success"])
                all_success = all(successes)
                if all_success:
                    fallback_successes += 1
                    success_rate += 1
                    for i in range(num_agents):
                        steps.append(search_based[1][idx][i]["steps"])
                        rewards.append(search_based[1][idx][i]["rewards"])
                        cumulative_costs.append(
                            search_based[1][idx][i]["cumulative_costs"]
                        )

                    mean_steps.append(np.mean(steps))
                    mean_rewards.append(np.mean(rewards))
                    mean_cumulative_costs.append(np.max(cumulative_costs))
            continue
        for i in range(num_agents):
            successes.append(record[i]["success"])
        all_success = all(successes)
        if all_success:
            success_rate += 1
            for i in range(num_agents):
                steps.append(record[i]["steps"])
                rewards.append(record[i]["rewards"])
                cumulative_costs.append(record[i]["cumulative_costs"])

            mean_steps.append(np.mean(steps))
            mean_rewards.append(np.mean(rewards))
            mean_cumulative_costs.append(np.max(cumulative_costs))

    metrics = {
        "mean_steps": mean_steps,
        "mean_rewards": mean_rewards,
        "mean_cumulative_costs": mean_cumulative_costs,
        "success_rate": success_rate / len(records),
    }

    return metrics


def means_and_stddevs(metrics, num_agents):
    key = "mean_cumulative_costs" if num_agents > 1 else "cumulative_costs"
    unconstrained_cc = metrics[0][key]
    unconstrained_search_cc = metrics[1][key]
    constrained_cc = metrics[2][key]
    constrained_search_cc = [metric[key] for metric in metrics[3]]

    unconstrained_cc_mean = np.mean(unconstrained_cc)
    unconstrained_search_cc_mean = np.mean(unconstrained_search_cc)
    constrained_cc_mean = np.mean(constrained_cc)
    constrained_search_cc_means = [np.mean(metric) for metric in constrained_search_cc]
    best_constrained_search_cc_index = np.argmin(constrained_search_cc_means)

    unconstrained_cc_stddev = np.std(unconstrained_cc)
    unconstrained_search_cc_stddev = np.std(unconstrained_search_cc)
    constrained_cc_stddev = np.std(constrained_cc)
    constrained_search_cc_stddevs = [np.std(metric) for metric in constrained_search_cc]

    unconstrained_sr = metrics[0]["success_rate"]
    unconstrained_search_sr = metrics[1]["success_rate"]
    constrained_sr = metrics[2]["success_rate"]
    constrained_search_sr = [metric["success_rate"] for metric in metrics[3]]

    edge_cost_factors = [0.1, 0.25, 0.5, 0.75, 1.0]

    print(f"Num Agents\t\t\t: {num_agents}")
    print(
        f"Unconstrained\t\t\t: {unconstrained_cc_mean:.2f} +/- {unconstrained_cc_stddev:.2f} "
        f"({unconstrained_sr:.2f})"
    )
    print(
        f"Unconstrained Search\t\t: {unconstrained_search_cc_mean:.2f} +/- {unconstrained_search_cc_stddev:.2f} "
        f"({unconstrained_search_sr:.2f})"
    )
    print(
        f"Constrained\t\t\t: {constrained_cc_mean:.2f} +/- {constrained_cc_stddev:.2f} ({constrained_sr:.2f})"
    )
    for idx, metric in enumerate(constrained_search_cc_means):
        print(
            f"Constrained Search ({edge_cost_factors[idx]})\t: "
            f"{metric:.2f} +/- {constrained_search_cc_stddevs[idx]:.2f} ({constrained_search_sr[idx]:.2f})"
        )
    print(
        f"Best Constrained Search\t\t: {constrained_search_cc_means[best_constrained_search_cc_index]:.2f} "
        f"+/- {constrained_search_cc_stddevs[best_constrained_search_cc_index]:.2f} "
        f"({constrained_search_sr[best_constrained_search_cc_index]:.2f})"
    )


def collect_metrics(basedir, problem_type, n_agents):

    if n_agents == 1:
        local_basedir = basedir / "single_agent" / problem_type
        unconstrained_records = np.load(
            local_basedir / "unconstrained_records.npy", allow_pickle=True
        )
        unconstrained_search_records = np.load(
            local_basedir / "unconstrained_search_records.npy", allow_pickle=True
        )
        constrained_records = np.load(
            local_basedir / "constrained_records.npy", allow_pickle=True
        )
        constrained_search_factored_records_cc = np.load(
            local_basedir / "constrained_search_factored_records.npy", allow_pickle=True
        )

        unconstrained_metrics = extract_single_agent_metrics(unconstrained_records)
        unconstrained_search_metrics = extract_single_agent_metrics(
            unconstrained_search_records, (True, unconstrained_records)
        )
        constrained_metrics = extract_single_agent_metrics(constrained_records)
        constrained_search_factored_metrics_cc = [
            extract_single_agent_metrics(csr, (True, constrained_records))
            for csr in constrained_search_factored_records_cc
        ]
    else:
        local_basedir = basedir / "multi_agent" / problem_type

        unconstrained_records = np.load(
            local_basedir / f"unconstrained_records_{n_agents}.npy", allow_pickle=True
        )
        unconstrained_search_records = np.load(
            local_basedir / f"unconstrained_search_records_{n_agents}.npy",
            allow_pickle=True,
        )
        constrained_records = np.load(
            local_basedir / f"constrained_records_{n_agents}.npy", allow_pickle=True
        )
        constrained_search_factored_records_cc = np.load(
            local_basedir / f"constrained_search_factored_records_{n_agents}.npy",
            allow_pickle=True,
        )

        unconstrained_metrics = extract_multi_agent_metrics(
            unconstrained_records, n_agents
        )
        unconstrained_search_metrics = extract_multi_agent_metrics(
            unconstrained_search_records, n_agents, (True, unconstrained_records)
        )
        constrained_metrics = extract_multi_agent_metrics(constrained_records, n_agents)
        constrained_search_factored_metrics_cc = [
            extract_multi_agent_metrics(csr, n_agents, (True, constrained_records))
            for csr in constrained_search_factored_records_cc
        ]

    return means_and_stddevs(
        [
            unconstrained_metrics,
            unconstrained_search_metrics,
            constrained_metrics,
            constrained_search_factored_metrics_cc,
        ],
        n_agents,
    )


if __name__ == "__main__":

    num_agents = [1, 5, 10, 20]
    problem_types = ["easy", "medium", "hard"]
    env_types = [
        "centerdot",
        "sc2_staging_08",
        "sc0_staging_20",
        "sc3_staging_05",
        "sc3_staging_11",
        "sc3_staging_15",
    ]

    for env_type in env_types:
        print("*" * 50)
        print(f"\tEnvironment Type: {env_type}")
        print("*" * 50)
        basedir = Path("pud/plots/data/" + env_type)
        if "staging" in env_type:
            num_agents = [1, 5, 10]
        for problem_type in problem_types:
            for n_agent in num_agents:
                print("-" * 50)
                print(f"\tProblem Type: {problem_type}")
                print("-" * 50)
                try:
                    collect_metrics(basedir, problem_type, n_agent)
                except FileNotFoundError:
                    continue
                print()
            print("\n\n")
