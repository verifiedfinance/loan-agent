"""Writes and executes notebooks/headroom_paper.ipynb (the paper's figures and LaTeX tables, with the narrative)."""
import nbformat as nbf
from nbclient import NotebookClient

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# A small in-house model that answers covenant-headroom questions: SFT, then RL with a verifiable reward

**Story.** A lender has to answer, every quarter and for every loan: *is the borrower in compliance, how far can its
earnings fall, and how much more debt can it take before a covenant is breached?* Borrower figures are confidential and
decisions must be auditable, so the system should run in-house. We train an open 7B model (Qwen2.5-Coder-7B + LoRA) to
turn each covenant into an SMT formula over the certificate's line items; the Z3 solver then answers all three questions
exactly. Supervised training on 789 teacher pairs gives a usable formalizer; reinforcement learning with a label-free,
solver-checked reward that also enforces the formula interface lifts it further, above open models 4-5x larger that read
the clause directly. Formulas can be proven correct, and two independent
formulas that Z3 proves equivalent are almost always right: a label-free self-check that decides which covenants run
automatically.

Every number below is computed from raw result files by `src/headroom/paper.py`. Tables are written as booktabs LaTeX
(`results/paper/tables/*.tex`, package: `booktabs`) and shown rendered; figures follow the figures4papers house style
and are saved as PDF and PNG (`results/paper/figures/`). Pre-registration and every later change: `docs/headroom_PLAN.md`.
""")
code(r"""
import sys
from pathlib import Path
SRC = Path.cwd().parent / 'src'
sys.path[:0] = [str(SRC)] + [str(d) for d in sorted(SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))]
import paper
from IPython.display import Image, Markdown, display

def show_table(fn):
    latex, markdown = fn()[:2]
    display(Markdown(markdown))
    print(latex)

def show_fig(fn, width=1100):
    display(Image(filename=str(fn()), width=width))
""")

md("## 0. Overview")
code("show_fig(paper.fig_overview_tex, 1400)")
md(r"""
**Figure 0.** (a) Every quarter, for every loan, an analyst needs a covenant's status, earnings cushion and debt capacity
from the clause and the compliance certificate. (b) A real benchmark case (borrower 1% short of its Total Debt / EBITDA
limit): a 7B model reading the clause directly gets it wrong; our in-house 7B formalizer writes a formula once, and Z3
answers exactly. Formulas are proven once and reused every quarter. (c) Earnings-cushion accuracy on borrowers never seen
in training. The figure is TikZ/pgfplots (`results/paper/figures/fig0_overview.tex`), with every number filled in from
the result files.
""")

md("## Method")
code("show_fig(paper.fig_method_tex, 1300)")
md(r"""
**Figure M.** Stage I: teacher formulas filtered by Z3 and a semantic review train the 7B model with SFT. Stage II: GRPO
on unlabeled clauses; Z3 scores each sampled formula (0 if the parameter interface breaks, else 0.3 + 0.7 x the share of
threshold tests passed). Stage III: in-house, formulas proven equal by two formalizers run every quarter; the rest go to
an analyst.
""")

md("## 1. The benchmark\nBorrower-disjoint clauses, two-reader reference readings, constructed certificates, and a 100% reference validation through the same Z3 code the formalizers use.")
code("show_table(paper.table_benchmark)")

md("## 2. Main result")
code("show_fig(paper.fig_main, 1300)")
md(r"""
**Figure 1.** Share of questions answered correctly on 1,064 cases (143 clauses). Hatched red: open models that read the
clause and answer directly; blue: our 7B formalizers with Z3. The error bar on SFT + RL is one standard deviation over
three seeds.
""")
code("show_table(paper.table_main)")
code("show_table(paper.table_paired)")
md(r"""
**Reading.** RL with the interface-enforcing reward adds about 9 points over SFT on every seed, and the resulting 7B
formalizer beats Qwen3.8-27B (+6.3, p_Holm = 0.004) and Qwen2.5-Coder-32B (+9.2, p_Holm < 0.001) reading the clause
directly. The first RL run, without the interface term, often dropped required parameters and
fell to 58.4; it is reported, not hidden (Table 2).
""")

md("## 3. Size")
code("show_fig(paper.fig_size, 800)")
md("**Figure 2.** Per-clause score against the size of the answering model; open readers with 95% clause-bootstrap intervals, and our two 7B formalizers.")

md("## 4. Verification: proofs instead of samples")
code("show_fig(paper.fig_verification, 1300)")
md(r"""
**Figure 4.** (a) Z3 proves each formula equivalent to the reference reading for every input, finds a counterexample, or
the formula is unusable. (b) Without any reference: two formalizers write formulas for the same clause and Z3 checks
whether they agree for every input. Agreed clauses run automatically; the rest go to an analyst.
""")
code("show_table(paper.table_verification)")
code("show_table(paper.table_selfcheck)")

md("## 5. Training")
code("show_fig(paper.fig_training, 1100)")
md(r"""
**Figure 3.** Learning curves. (a) SFT loss over one epoch on 789 pairs. (b-d) GRPO, three seeds: (b) reward; (c) share
of steps whose four samples differ in reward, the only steps with a gradient (95.5 ± 0.6% of steps carry none); (d) share
of samples that keep the required parameter interface. A few dozen informative steps move the model by 9 points.
""")

md("## 6. Where it helps, and where it does not")
code("show_fig(paper.fig_cases, 1300)")
md(r"""
**Figure 5.** Per-clause score by case type (cushion placed below, near and well above the limit; a needed figure
missing) and by clause type. RL was trained only on clauses without date schedules or conditions: on those it reaches
about 99%, while on the 27 clauses with schedules or conditions it does not improve on SFT.
""")
code("show_table(paper.table_sensitivity)")

md(r"""
## 7. Claims and limits

**Supported.** (1) SFT then RL with a solver-checked, interface-enforcing reward improves a 7B formalizer by about 9
points on held-out borrowers (three seeds). (2) The 7B formalizer answers headroom questions better than 27B and 32B open
models reading directly. (3) Formalizing beats reading at equal size
by about 30 points, mostly on the earnings cushion. (4) 71% of RL formulas are proven correct for every input; two
independent formulas that Z3 proves equivalent are right 98-100% of the time, covering 80-92% of clauses.

**Not claimed.** Real-world compliance accuracy (figures are constructed, references are two
model readings checked by Z3, not expert labels); clauses with date schedules or conditions (RL does not help there yet).
RL v2 is the second RL run, designed after the first one failed; its comparisons were fixed before training.
""")
code(r"""
import os
for d in (paper.FIG, paper.TAB):
    print(d.relative_to(paper.P.ROOT)); [print('   ', f) for f in sorted(os.listdir(d))]
""")

nb = nbf.v4.new_notebook()
nb['cells'] = C
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
from pathlib import Path
nbdir = str(Path(__file__).resolve().parents[1] / 'notebooks')
NotebookClient(nb, timeout=3000, kernel_name='python3', resources={'metadata': {'path': nbdir}}).execute()
nbf.write(nb, nbdir + '/headroom_paper.ipynb')
print('errors', sum(1 for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.output_type == 'error'))
