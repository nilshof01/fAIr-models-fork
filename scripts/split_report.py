"""Human-readable statistics and plots for a split, written next to a run.

Every training writes these into its output folder so the composition behind a
number is recoverable without re-deriving anything.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["train", "val", "test", "holdout"]
# colour-blind safe, one hue per role, assigned in fixed order
ROLE_COLOR = {"train": "#4269d0", "val": "#efb118", "test": "#ff725c",
              "holdout": "#6cc5b0"}


class SplitStats:
    """Per-split composition, as a table you can read and a frame you can save."""

    def __init__(self, df, role_col="role"):
        self.df = df
        self.role_col = role_col

    @property
    def roles(self):
        seen = list(dict.fromkeys(self.df[self.role_col]))
        known = [r for r in ORDER if r in seen]
        return known + sorted(r for r in seen if r not in ORDER)

    @property
    def table(self):
        rows = []
        for role in self.roles:
            g = self.df[self.df[self.role_col] == role]
            if not len(g):
                continue
            pop = g[~g.is_empty]
            rows.append({
                "split": role,
                "chips": len(g),
                "populated": len(pop),
                "n_empty": int(g.is_empty.sum()),
                "pct_empty": round(100 * g.is_empty.mean(), 1),
                "instances": int(g.n_instances.sum()),
                "inst_per_populated": round(pop.n_instances.mean(), 1) if len(pop) else 0.0,
                "median_density": float(pop.n_instances.median()) if len(pop) else 0.0,
                "max_density": int(g.n_instances.fillna(0).max()) if len(g) else 0,
                "blocks": g.block.nunique(),
                "countries": g.country.nunique(),
                "projects": g.project_name.nunique(),
            })
        t = pd.DataFrame(rows)
        # `empty`, `pop`, `gt`, `size` are DataFrame attributes: a column named
        # after one makes t.<name> return the method, silently, and plots or
        # sums built on it are wrong rather than broken.
        clash = set(t.columns) & set(dir(pd.DataFrame))
        assert not clash, f"column shadows a DataFrame attribute: {clash}"
        return t

    @property
    def country_table(self):
        p = self.df.pivot_table(index="country", columns=self.role_col,
                                values="tile_id", aggfunc="size", fill_value=0)
        cols = [c for c in self.roles if c in p.columns]
        p = p[cols]
        return p.loc[p.sum(axis=1).sort_values(ascending=False).index]

    def concentration(self):
        """How much of each split's building mass sits in its biggest chip."""
        rows = []
        for role in self.roles:
            g = self.df[self.df[self.role_col] == role]
            if not len(g) or g.n_instances.sum() == 0:
                continue
            ni = g.n_instances.fillna(0)
            tot = ni.sum()
            rows.append({"split": role,
                         "top_chip_pct": round(100 * ni.max() / tot, 1),
                         "top10_chip_pct": round(100 * ni.nlargest(10).sum() / tot, 1),
                         "top_project_pct": round(
                             100 * g.groupby("project_name").n_instances.sum().max() / tot, 1)})
        return pd.DataFrame(rows)

    def text(self, title="split composition"):
        lines = [f"== {title} ==", self.table.to_string(index=False), ""]
        lines += ["-- building-mass concentration (a high number means the metric "
                  "is dominated by a few chips) --",
                  self.concentration().to_string(index=False), ""]
        ct = self.country_table
        lines += [f"-- chips by country ({len(ct)} countries) --",
                  ct.head(15).to_string()]
        if len(ct) > 15:
            lines.append(f"   ... {len(ct)-15} more")
        return "\n".join(lines)

    def save(self, out_dir, prefix="split"):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(out / f"{prefix}_stats.csv", index=False)
        self.country_table.to_csv(out / f"{prefix}_by_country.csv")
        (out / f"{prefix}_stats.txt").write_text(self.text())
        return out / f"{prefix}_stats.csv"

    def plot(self, out_dir, prefix="split", title=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = self.table
        roles = list(t.split)
        colors = [ROLE_COLOR.get(r, "#9498a0") for r in roles]
        fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
        fig.suptitle(title or "Split composition", fontsize=13)

        a = ax[0, 0]
        # empty chips get their own neutral fill, not a faint tint of the same
        # hue - at low alpha the segment is invisible and the bar reads wrong
        a.bar(roles, t["populated"], color=colors, label="populated")
        a.bar(roles, t["n_empty"], bottom=t["populated"], color="#e6e8ec",
              edgecolor=colors, linewidth=1.2, label="verified empty")
        for i, r in t.iterrows():
            a.text(i, r.chips, f"{int(r.chips):,}\n{r.pct_empty:.0f}% empty",
                   ha="center", va="bottom", fontsize=8)
        a.set_title("chips", fontsize=10)
        a.set_ylim(0, t["chips"].max() * 1.30)
        a.legend(fontsize=7, loc="upper right", framealpha=0.9)
        a.spines[["top", "right"]].set_visible(False)

        a = ax[0, 1]
        a.bar(roles, t["instances"], color=colors)
        for i, r in t.iterrows():
            a.text(i, r.instances, f"{int(r.instances):,}", ha="center",
                   va="bottom", fontsize=8)
        a.set_title("building instances", fontsize=10)
        a.set_ylim(0, t["instances"].max() * 1.18)
        a.spines[["top", "right"]].set_visible(False)

        a = ax[1, 0]
        data = [self.df[(self.df[self.role_col] == r) & (~self.df.is_empty)]
                .n_instances.values for r in roles]
        bp = a.boxplot(data, tick_labels=roles, showfliers=False,
                       patch_artist=True, medianprops=dict(color="#22252a"))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        a.set_title("buildings per populated chip", fontsize=10)
        a.spines[["top", "right"]].set_visible(False)

        a = ax[1, 1]
        ct = self.country_table.head(8)
        share = ct.div(ct.sum(axis=0), axis=1) * 100
        bottom = np.zeros(len(share.columns))
        for country in share.index:
            a.bar(share.columns, share.loc[country], bottom=bottom, label=country)
            bottom += share.loc[country].values
        a.set_title("country mix (%, top 8)", fontsize=10)
        a.set_ylim(0, 100)
        a.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5),
                 frameon=False)
        a.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        p = Path(out_dir) / f"{prefix}_composition.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        return p
