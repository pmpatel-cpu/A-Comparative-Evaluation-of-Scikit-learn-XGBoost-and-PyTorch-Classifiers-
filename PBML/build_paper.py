"""
build_paper.py
--------------
Builds the research paper (.docx) from results.json produced by run_experiments.py.
Every number, table and results sentence is computed from results.json - nothing is typed in by hand.

Usage:
    pip install python-docx matplotlib numpy scipy
    python build_paper.py --results results.json --out Research_Paper.docx
"""
import argparse, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

TITLE = ("A Comparative Evaluation of Scikit-learn, XGBoost and PyTorch Classifiers on Diagnostic "
         "and Image Benchmarks Using Repeated Cross-Validation")
AUTHOR = "Patel Priyankakumari Miteshkumar"
AFFIL = "Sabarmati University, Ahmedabad, India"
EMAIL = "pmpatel@amrolicollege.ac.in"
KEYWORDS = ("Python; Machine Learning; Deep Learning; Scikit-learn; XGBoost; PyTorch; "
            "Repeated Cross-Validation; Model Comparison; Breast Cancer Wisconsin; Handwritten Digits")

REFERENCES = [
    "Alpaydin, E., & Kaynak, C. (1998). Optical recognition of handwritten digits [Data set]. UCI Machine Learning Repository.",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785",
    "Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics, 21, 6.",
    "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.",
    "Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1–30.",
    "Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems, 35 (Datasets and Benchmarks Track).",
    "Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65–70.",
    "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations.",
    "Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. Proceedings of the 14th International Joint Conference on Artificial Intelligence, 1137–1143.",
    "Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52(3), 239–281.",
    "Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. Information Fusion, 81, 84–90.",
    "Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. Proceedings of IS&T/SPIE International Symposium on Electronic Imaging, 1905, 861–870.",
    "Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics, 7, 91.",
    "Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast Cancer Wisconsin (Diagnostic) [Data set]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
]


# ------------------------------------------------------------------ docx helpers
def set_cell_bg(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
        run._r.append(el)


class Paper:
    def __init__(self):
        self.d = Document()
        s = self.d.sections[0]
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)
        st = self.d.styles["Normal"]
        st.font.name = "Times New Roman"; st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        add_page_number(s)

    def p(self, text="", bold=False, italic=False, align="justify", size=None, after=6, highlight=False, keep=False):
        para = self.d.add_paragraph()
        para.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                          "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
        para.paragraph_format.space_after = Pt(after)
        para.paragraph_format.keep_with_next = keep
        r = para.add_run(text); r.bold = bold; r.italic = italic
        if size: r.font.size = Pt(size)
        if highlight: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return para

    def h1(self, t):
        para = self.p(t, bold=True, size=13, align="left", after=4, keep=True)
        para.paragraph_format.space_before = Pt(12)

    def h2(self, t):
        para = self.p(t, bold=True, italic=True, size=11.5, align="left", after=3, keep=True)
        para.paragraph_format.space_before = Pt(8)

    def bullets(self, items):
        for it in items:
            para = self.d.add_paragraph(style="List Bullet")
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            para.paragraph_format.space_after = Pt(2)
            para.add_run(it)

    def table(self, caption, header, rows, widths=None, first_left=True):
        self.p(caption, bold=True, align="left", size=10, after=3, keep=True)
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]; c.text = ""
            r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(9)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell_bg(c, "D9E2F3")
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(9)
                cells[i].paragraphs[0].alignment = (WD_ALIGN_PARAGRAPH.LEFT if (i == 0 and first_left)
                                                    else WD_ALIGN_PARAGRAPH.CENTER)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)


# ------------------------------------------------------------------ result helpers
def pm(s, scale=100, nd=2):
    return f"{scale * s['mean']:.{nd}f} ± {scale * s['std']:.{nd}f}"


def short(name):
    return name.replace(" (scikit-learn)", " (sklearn)").replace(" (PyTorch)", " (PyTorch)")


def join_and(items):
    items = list(items)
    if not items: return ""
    if len(items) == 1: return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def nb_test(res, a, b, n, splits):
    A = np.array([f["acc"] for f in res[a]["folds"]]); B = np.array([f["acc"] for f in res[b]["folds"]])
    d = A - B; k = len(d); n_test = n / splits; n_train = n - n_test
    var = d.var(ddof=1)
    if var == 0: return float(d.mean()), 1.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    return float(d.mean()), float(2 * stats.t.sf(abs(t), k - 1))


def p_str(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def make_figure(R, path):
    names = list(R["datasets"])
    fig, axes = plt.subplots(1, len(names), figsize=(11, 4.2))
    axes = np.atleast_1d(axes)
    for ax, dn in zip(axes, names):
        ms = R["datasets"][dn]["models"]
        data = [[100 * f["acc"] for f in ms[m]["folds"]] for m in ms]
        ax.boxplot(data, showmeans=True)
        ax.set_xticklabels([short(m).replace(" ", "\n", 1) for m in ms])
        ax.set_title(dn, fontsize=10); ax.set_ylabel("Accuracy per test fold (%)")
        ax.tick_params(axis="x", labelsize=7.5); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------------------------ main builder
def build(R, out, fig_path):
    cfg, env, DS = R["config"], R["env"], R["datasets"]
    k, rep = cfg["splits"], cfg["repeats"]; nfolds = k * rep
    dnames = list(DS); bc, dg = dnames[0], dnames[1]
    model_names = list(DS[bc]["models"])
    not_run = R.get("not_run", [])
    P = Paper()

    # ---- title block
    P.p(TITLE, bold=True, size=16, align="center", after=6)
    P.p(AUTHOR, bold=True, align="center", after=0)
    P.p(AFFIL, italic=True, align="center", after=0)
    P.p(EMAIL, align="center", after=10)
    if not_run:
        miss = join_and(not_run)
        P.p("DRAFT NOTE – INCOMPLETE, NOT FOR SUBMISSION: the experiments for " + miss + " have NOT been run "
            "for this version, and these models are absent from every table and figure below. The results shown "
            "cover only the remaining models and must not be presented as a complete experimental comparison of "
            "Scikit-learn, XGBoost and PyTorch. Before submitting to a journal, run run_experiments.py on a machine "
            "with all libraries installed, rebuild the paper with build_paper.py, and re-check every statement "
            "against the complete results. Delete this note only after that.", bold=True, highlight=True)
        hp = P.d.sections[0].header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hr = hp.add_run("DRAFT – INCOMPLETE: " + miss + " results missing – not for submission")
        hr.bold = True; hr.font.size = Pt(9); hr.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    # ---- abstract
    best = {d: DS[d]["stats"]["best"] for d in dnames}
    def best_s(d):
        b = best[d]; return f"{short(b)} ({pm(DS[d]['models'][b]['summary']['acc'])}%)"
    def equiv(d):
        return [short(m) for m, v in DS[d]["stats"]["vs_best"].items() if v["p_holm"] >= 0.05]
    tt = [DS[d]["models"][m]["summary"]["train_time"]["mean"] for d in dnames for m in DS[d]["models"]]
    ratio = max(max(DS[d]["models"][m]["summary"]["train_time"]["mean"] for m in DS[d]["models"]) /
                min(DS[d]["models"][m]["summary"]["train_time"]["mean"] for m in DS[d]["models"]) for d in dnames)
    fw_run = list(dict.fromkeys(DS[bc]["models"][m]["framework"] for m in model_names))
    abs_ = (
        "Python offers many libraries for machine learning (ML) and deep learning (DL), but library and model "
        "choices are often justified with single train-test splits that cannot separate genuine differences from "
        f"sampling noise. This study compares {len(model_names)} classifiers – {join_and([short(m) for m in model_names])} – "
        f"implemented with {join_and(fw_run)} on two public benchmarks: the Breast Cancer Wisconsin "
        f"(Diagnostic) dataset ({DS[bc]['n']} records, {DS[bc]['features']} features, binary) and the Optical Recognition "
        f"of Handwritten Digits dataset ({DS[dg]['n']} images, {DS[dg]['features']} pixel features, "
        f"{DS[dg]['classes']} classes). Every model is evaluated with {rep} × {k}-fold repeated stratified "
        "cross-validation ({0} test folds), with hyperparameters selected by inner cross-validation, and compared "
        "using accuracy, precision, recall, F1-score, Matthews correlation coefficient, ROC-AUC, training time and "
        "inference time. Differences are assessed with the Friedman test and Nadeau–Bengio corrected resampled "
        "t-tests with Holm adjustment. ".format(nfolds)
        + f"The highest mean accuracy was obtained by {best_s(bc)} on the breast-cancer data and by {best_s(dg)} "
        f"on the digits data; "
        + (f"on the breast-cancer data {join_and(equiv(bc))} could not be statistically distinguished from the best model"
           if equiv(bc) else "on the breast-cancer data every other model was significantly worse than the best")
        + (f", and on the digits data {join_and(equiv(dg))} likewise" if equiv(dg) else
           ", while on the digits data every other model was significantly worse than the best")
        + ". " + ("The results show that accuracy gaps between the leading models are often not statistically "
                  "distinguishable, which single-split rankings cannot reveal. " if (equiv(bc) or equiv(dg)) else
                  "The results show statistically supported accuracy differences among the models. ")
        + f"Mean training time differed by a factor of up to {ratio:.0f} between the fastest and slowest model.")
    P.p("Abstract", bold=True, align="left", after=2)
    P.p(abs_)
    P.p("Keywords: " + KEYWORDS, italic=True, after=8)

    # ---- 1 Introduction
    P.h1("1. Introduction")
    P.p("Python has become the dominant language for applied machine learning, largely because of a mature ecosystem of "
        "libraries. Scikit-learn provides a unified interface to classical algorithms (Pedregosa et al., 2011), XGBoost "
        "provides an optimised implementation of gradient-boosted decision trees (Chen & Guestrin, 2016), and PyTorch "
        "provides tensor computation and automatic differentiation for neural networks (Paszke et al., 2019). "
        "Practitioners and students frequently have to choose among these tools for a new classification problem.")
    P.p("Published comparisons often rely on a single random train-test split of a small dataset. On a test set of "
        "about one hundred samples, one additional correct prediction changes accuracy by close to one percentage "
        "point, so rankings based on one split can reverse under a different random seed. Reliable comparison needs "
        "repeated resampling, hyperparameter selection that does not leak test information (Varma & Simon, 2006), "
        "and inference that accounts for the overlap between cross-validation training sets (Nadeau & Bengio, 2003).")
    P.p("This paper therefore asks the following research questions:")
    P.bullets([
        "RQ1: How do classical ML models, gradient-boosted trees and neural networks implemented in Scikit-learn, "
        "XGBoost and PyTorch compare in predictive performance on a small tabular diagnostic dataset and on a "
        "small image-based multi-class dataset?",
        "RQ2: Which of the observed performance differences are statistically distinguishable once resampling "
        "variance and multiple comparisons are taken into account?",
        "RQ3: How do training and inference times differ across the approaches?",
        "RQ4: When the same multi-layer perceptron is implemented in two libraries (Scikit-learn and PyTorch) "
        "with matched hyperparameters, do the libraries yield different results?",
    ])
    P.p("The contributions are: (i) a reproducible repeated-cross-validation protocol with inner-loop tuning and "
        "leakage-free preprocessing; (ii) statistical comparison using the Friedman test and corrected resampled "
        "t-tests; (iii) a controlled same-algorithm comparison of two MLP implementations; and (iv) released code "
        "that regenerates every table in this paper.")

    # ---- 2 Related work
    P.h1("2. Related Work")
    P.h2("2.1 Libraries and algorithms")
    P.p("Scikit-learn offers a consistent API for preprocessing, model selection and evaluation of classical methods "
        "such as logistic regression, support vector machines (Cortes & Vapnik, 1995) and random forests (Breiman, "
        "2001). XGBoost implements regularised gradient boosting with efficient tree construction and became a "
        "standard choice for structured data (Chen & Guestrin, 2016). PyTorch offers flexible, define-by-run neural "
        "network construction trained with optimisers such as Adam (Kingma & Ba, 2015; Paszke et al., 2019).")
    P.h2("2.2 Tree-based models versus neural networks on tabular data")
    P.p("Shwartz-Ziv and Armon (2022) compared recent deep tabular models with XGBoost across several datasets and "
        "reported that XGBoost generally performed better and required less tuning, while ensembles of deep models "
        "and XGBoost performed best. Grinsztajn et al. (2022) similarly found that tree-based models remain "
        "state of the art on medium-sized tabular datasets. These studies motivate including both tree ensembles and "
        "neural networks in a comparison, but they concern larger benchmark collections than the small datasets "
        "studied here.")
    P.h2("2.3 Evaluation methodology")
    P.p("Cross-validation is the standard tool for estimating generalisation error (Kohavi, 1995), but using the same "
        "folds for tuning and for reporting yields optimistic estimates (Varma & Simon, 2006), which motivates "
        "tuning inside each training fold. Because cross-validation training sets overlap, the ordinary paired t-test "
        "is too liberal; Nadeau and Bengio (2003) proposed a corrected resampled t-test. For comparison of several "
        "classifiers, Demšar (2006) recommends non-parametric omnibus tests such as the Friedman test followed by "
        "corrected pairwise tests. For binary medical-style tasks, the Matthews correlation coefficient is a more "
        "informative single summary than accuracy or F1 when classes are imbalanced (Chicco & Jurman, 2020).")
    P.h2("2.4 Research gap")
    P.p("Many practical comparisons of Python libraries report one split, untuned baselines and no significance "
        "testing. This study applies the methodological safeguards above to a small, reproducible set of "
        "benchmarks that anyone can regenerate without external downloads.")

    # ---- 3 Methodology
    P.h1("3. Methodology")
    P.h2("3.1 Datasets")
    bcc = DS[bc]["class_counts"]; dgc = DS[dg]["class_counts"]
    P.p("Two datasets distributed with Scikit-learn are used. The Breast Cancer Wisconsin (Diagnostic) dataset "
        "(Street et al., 1993; Wolberg et al., 1995) contains numerical features computed from digitised images of "
        f"fine-needle aspirates of breast masses; it has {DS[bc]['n']} records, {DS[bc]['features']} features, "
        f"{bcc.get('0', bcc.get(0))} malignant and {bcc.get('1', bcc.get(1))} benign cases. Malignant is treated as the positive class, so recall "
        "corresponds to sensitivity to malignancy. The Optical Recognition of Handwritten Digits dataset "
        f"(Alpaydin & Kaynak, 1998) contains {DS[dg]['n']} images of handwritten digits reduced to 8 × 8 pixels "
        f"({DS[dg]['features']} integer features), with {DS[dg]['classes']} roughly balanced classes. "
        "The second dataset adds a multi-class, image-derived problem so that conclusions are not confined to one "
        "data type.")
    P.table("Table 1. Dataset summary.",
            ["Dataset", "Records", "Features", "Classes", "Task"],
            [[bc, DS[bc]["n"], DS[bc]["features"], DS[bc]["classes"], "Binary classification"],
             [dg, DS[dg]["n"], DS[dg]["features"], DS[dg]["classes"], "Multi-class classification"]],
            widths=[2.4, 0.8, 0.8, 0.7, 1.8])

    P.h2("3.2 Models")
    P.p("Table 2 lists the models and hyperparameters. Models with a grid are tuned by 3-fold stratified "
        "cross-validation on the training portion of each outer fold, using accuracy as the selection criterion, "
        "and then refitted on the whole training portion. The two MLPs are deliberately given identical, fixed "
        "settings (two hidden layers of 64 and 32 ReLU units, Adam with learning rate 10⁻³, batch size 32, 100 epochs, "
        "no dropout, no weight decay, seed 42) so that any difference between them reflects the library "
        "implementation (for example weight initialisation and training loop) rather than configuration. "
        "The MLPs are not tuned; this is an intentional design choice for the controlled comparison and a limitation "
        "for the tuned-versus-untuned comparison (Section 7).")
    P.table("Table 2. Models, libraries and hyperparameters.",
            ["Model", "Library", "Fixed settings", "Tuning grid (inner 3-fold CV)"],
            [["Logistic Regression", "Scikit-learn", "L2 penalty, max_iter = 2000", "C ∈ {0.1, 1, 10}"],
             ["SVM (RBF)", "Scikit-learn", "RBF kernel, probability = True", "C ∈ {0.1, 1, 10}; γ ∈ {scale, 0.01}"],
             ["Random Forest", "Scikit-learn", "200 trees", "max_depth ∈ {None, 10}; min_samples_leaf ∈ {1, 3}"],
             ["XGBoost", "XGBoost", "200 boosting rounds, log-loss", "max_depth ∈ {3, 5}; learning_rate ∈ {0.05, 0.1}"],
             ["MLP (sklearn)", "Scikit-learn", "64–32 ReLU, Adam 10⁻³, batch 32, 100 epochs, α = 0", "none (fixed)"],
             ["MLP (PyTorch)", "PyTorch", "64–32 ReLU, Adam 10⁻³, batch 32, 100 epochs, cross-entropy", "none (fixed)"]],
            widths=[1.4, 1.0, 2.3, 1.9])

    P.h2("3.3 Preprocessing and evaluation protocol")
    P.p(f"Features are standardised (zero mean, unit variance) for logistic regression, SVM and both MLPs; "
        "tree ensembles are invariant to monotone feature scaling and receive raw features. Standardisation is "
        "implemented inside a pipeline so it is fitted on training data only in every fold, which prevents "
        f"information leakage. Evaluation uses {rep} repetitions of stratified {k}-fold cross-validation "
        f"(random seed 42), giving {nfolds} test folds per dataset. All models see identical folds.")
    P.h2("3.4 Metrics")
    P.p("Accuracy, precision, recall and F1-score are computed on each test fold. For the binary dataset these "
        "refer to the malignant class; for the multi-class dataset they are macro-averaged. The Matthews correlation "
        "coefficient (MCC) and the area under the ROC curve (ROC-AUC; one-versus-rest macro average for the "
        "multi-class case) are also reported. For the binary dataset the number of false negatives (malignant "
        "cases predicted benign) per test fold is reported because missed malignancies are the costly error type. "
        "Training time is the time to refit the selected configuration on the outer training portion (grid search "
        "time is excluded); inference time is the time to predict labels for one test fold. SVM probability "
        "estimates are obtained by Scikit-learn's internal cross-validated calibration, which inflates its "
        "training time relative to a plain SVM.")
    P.h2("3.5 Statistical analysis")
    P.p("For each dataset, per-fold accuracies of all models are compared with the Friedman test (Demšar, 2006). "
        "The model with the highest mean accuracy is then compared with each other model using the corrected "
        "resampled t-test of Nadeau and Bengio (2003), with the variance term scaled by "
        f"(1/{nfolds} + n_test/n_train), and p-values are adjusted with Holm’s method (Holm, 1979). "
        "Because cross-validation folds are not independent, p-values are approximate and are interpreted as "
        "evidence strength rather than exact error rates. A significance level of 0.05 is used.")

    # ---- 4 Experimental setup
    P.h1("4. Experimental Setup")
    libs = [f"Python {env['python']}", f"scikit-learn {env['sklearn']}", f"NumPy {env['numpy']}", f"SciPy {env['scipy']}"]
    if env.get("xgboost"): libs.append(f"XGBoost {env['xgboost']}")
    if env.get("torch"): libs.append(f"PyTorch {env['torch']}")
    P.p("All experiments were run on CPU. Software versions: " + ", ".join(libs) + ". Platform: "
        f"{env['platform']} ({env['processor']}). Models were run sequentially with n_jobs = 1. Timing results are "
        "specific to this environment. Code and instructions to regenerate this paper are provided with the "
        "supplementary material.")

    # ---- 5 Results
    P.h1("5. Results")
    if not_run:
        P.p("Note: results for " + join_and(not_run) + " are missing in this draft; the tables below are a partial "
            "comparison only.", bold=True, highlight=True)
    P.h2("5.1 Predictive performance")
    tno = 3
    for d in dnames:
        ms = DS[d]["models"]; binary = DS[d]["kind"] == "binary"
        hdr = ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "MCC", "ROC-AUC"]
        if binary: hdr.append("FN / fold")
        rows = []
        for m, v in ms.items():
            s = v["summary"]
            row = [short(m), pm(s["acc"]), pm(s["prec"]), pm(s["rec"]), pm(s["f1"]),
                   pm(s["mcc"], 1, 3), pm(s["auc"], 1, 4)]
            if binary: row.append(pm(s["fn"], 1, 2))
            rows.append(row)
        what = "malignant-class" if binary else "macro-averaged"
        P.table(f"Table {tno}. {d}: mean ± standard deviation over {nfolds} test folds ({what} precision, recall and F1).",
                hdr, rows, widths=[1.3, 1.0, 1.0, 0.95, 0.9, 0.85, 0.85] + ([0.75] if binary else []))
        tno += 1
    P.p(f"Tables 3 and 4 report performance averaged over {nfolds} test folds per dataset. Figure 1 shows the "
        "distribution of per-fold accuracy.")
    make_figure(R, fig_path)
    P.d.add_picture(fig_path, width=Inches(6.4))
    P.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P.p("Figure 1. Per-fold test accuracy of each model (box = interquartile range, line = median, "
        "triangle = mean).", italic=True, align="center", size=10)

    P.h2("5.2 Statistical comparison")
    for d in dnames:
        stt = DS[d]["stats"]; b = stt["best"]
        rows = [[short(m), f"{100 * v['mean_diff']:.2f}", p_str(v["p"]), p_str(v["p_holm"]),
                 "Yes" if v["p_holm"] < 0.05 else "No"] for m, v in stt["vs_best"].items()]
        fr = stt["friedman_p"]
        P.p(f"{d}: the Friedman test {'gave p ' + p_str(fr) if fr is not None else 'could not be computed'}"
            + (" (differences among models are unlikely to be due to chance alone)" if fr is not None and fr < 0.05
               else " (no overall evidence of a difference)" if fr is not None else "")
            + f". {short(b)} has the highest mean accuracy and is the reference in Table {tno}.", after=3)
        P.table(f"Table {tno}. {d}: corrected resampled t-tests of {short(b)} against each other model.",
                ["Compared model", "Mean accuracy gap (pp)", "p (unadjusted)", "p (Holm)", "Significant at 0.05"],
                rows, widths=[1.8, 1.4, 1.2, 1.0, 1.2])
        tno += 1

    mlp_pair = ("MLP (scikit-learn)", "MLP (PyTorch)")
    if all(m in model_names for m in mlp_pair):
        P.h2("5.3 Same algorithm, two libraries")
        parts = []
        for d in dnames:
            gap, p = nb_test(DS[d]["models"], mlp_pair[1], mlp_pair[0], DS[d]["n"], k)
            a1 = DS[d]["models"][mlp_pair[0]]["summary"]["acc"]; a2 = DS[d]["models"][mlp_pair[1]]["summary"]["acc"]
            parts.append(f"on {d} the scikit-learn MLP reached {pm(a1)}% and the PyTorch MLP {pm(a2)}% "
                         f"(PyTorch minus scikit-learn: {100 * gap:+.2f} percentage points, corrected t-test p {p_str(p)})")
        P.p("With matched architecture and hyperparameters, " + "; ".join(parts) + ".")

    P.h2("5.4 Computational cost")
    rows = []
    for m in model_names:
        r = [short(m)]
        for d in dnames:
            s = DS[d]["models"][m]["summary"]
            r += [f"{s['train_time']['mean']:.4f} ({s['train_time']['std']:.4f})",
                  f"{1000 * s['infer_time']['mean']:.2f} ({1000 * s['infer_time']['std']:.2f})"]
        rows.append(r)
    P.table(f"Table {tno}. Mean (SD) training time in seconds and inference time in milliseconds per test fold.",
            ["Model", "Train (s) – breast cancer", "Infer (ms) – breast cancer", "Train (s) – digits", "Infer (ms) – digits"],
            rows, widths=[1.5, 1.35, 1.35, 1.2, 1.2])
    time_tno = tno
    tno += 1
    def extreme(d, key, fn):
        ms = DS[d]["models"]; m = fn(ms, key=lambda x: ms[x]["summary"][key]["mean"]); return short(m), ms[m]["summary"][key]["mean"]
    fast = extreme(bc, "train_time", min); slow = extreme(bc, "train_time", max)
    fast2 = extreme(dg, "train_time", min); slow2 = extreme(dg, "train_time", max)
    P.p(f"On the breast-cancer data the shortest mean training time was {fast[1]:.4f} s ({fast[0]}) and the longest "
        f"{slow[1]:.4f} s ({slow[0]}); on the digits data they were {fast2[1]:.4f} s ({fast2[0]}) and {slow2[1]:.4f} s "
        f"({slow2[0]}). Training times exclude hyperparameter search, which is considerably more expensive for the "
        "tuned models.")

    # ---- 6 Discussion
    P.h1("6. Discussion")
    nb, nd_ = DS[bc]["n"], DS[dg]["n"]
    P.p(f"Answering RQ1 and RQ2: on the breast-cancer data the best mean accuracy was achieved by {short(best[bc])}, "
        f"and on the digits data by {short(best[dg])}. The key point is that these rankings are only partly "
        "supported statistically. "
        + (f"For the breast-cancer data, {join_and(equiv(bc))} could not be separated from the best model after Holm "
           "adjustment. " if equiv(bc) else "")
        + (f"For the digits data, {join_and(equiv(dg))} could not be separated from the best model. " if equiv(dg) else "")
        + "Differences of around one percentage point should therefore not be read as evidence that one library or "
        "algorithm is better in general.")
    P.p("The observed pattern should not be extrapolated to all problems. Both datasets are small and "
        "low-dimensional, a regime in which well-regularised linear and kernel models are often very competitive "
        "with neural networks, and in which untuned neural networks may be at a disadvantage or advantage depending "
        "on initialisation and training length. Larger or more complex data may change the ranking, as the tabular "
        "benchmark literature indicates (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022).")
    P.p("For the binary medical-style task, accuracy alone hides the error type that matters most. The false-negative "
        "column of Table 3 shows how many malignant cases each model missed per test fold on average; a model with "
        "slightly lower accuracy but fewer false negatives could be preferable in practice. This study is a "
        "methodological benchmark and does not support any clinical use of these models.")
    P.p(f"Answering RQ3: computational cost differs greatly between approaches (Table {time_tno}). Inference time is on the order of milliseconds "
        "per test fold for all models on these datasets, while training time is dominated by ensemble size, kernel "
        "probability calibration and the number of neural-network epochs. These timings are single-environment "
        "measurements and should be treated as indicative.")
    P.h2("6.1 Practical implications")
    P.p("For small tabular or small image-derived datasets, a tuned classical model is a sensible baseline against "
        "which neural networks should be judged, and a neural network should be adopted only if repeated validation "
        "shows a gain that is both statistically and practically meaningful. Library choice should consider ecosystem, deployment needs and "
        "team familiarity, since matched implementations of the same algorithm behaved similarly where "
        "Section 5.3 applies.")

    # ---- 7/8 limits & future
    P.h1("7. Threats to Validity")
    P.bullets([
        "Only two small datasets are used; conclusions do not necessarily transfer to large, noisy, or higher-dimensional data.",
        "The MLPs are not tuned while other models are, so the neural networks may be under-represented relative to their best achievable performance.",
        "Corrected resampled t-tests rely on approximations because cross-validation folds are dependent; p-values are indicative.",
        "Timing was measured in one CPU environment without repeated independent timing sessions; GPU training and memory usage were not measured.",
        "Only three libraries are compared; TensorFlow/Keras, LightGBM and CatBoost are not included.",
        "Tuning optimises accuracy only; other objectives (for example recall of malignant cases) could select different configurations.",
    ])
    P.h1("8. Future Work")
    P.p("Future work should add larger tabular datasets, a convolutional network for image data, additional libraries "
        "(TensorFlow/Keras, LightGBM, CatBoost), tuning of neural networks under an equal search budget, GPU and "
        "memory measurements, and calibration analysis for the medical-style task.")

    # ---- 9 Conclusion
    P.h1("9. Conclusion")
    P.p(f"This paper compared {len(model_names)} classifiers from Scikit-learn, XGBoost and PyTorch on a diagnostic "
        f"tabular dataset and a handwritten-digit image dataset using {rep} × {k}-fold repeated stratified "
        f"cross-validation with inner-loop tuning. {short(best[bc])} and {short(best[dg])} achieved the highest mean "
        "accuracy on the two datasets respectively. "
        + ("Some competing models were statistically indistinguishable from the leaders. " if (equiv(bc) or equiv(dg)) else
           "Every other model was significantly worse than the leader on its dataset. ")
        + f"Mean training time differed by a factor of up to {ratio:.0f} between the fastest and slowest model. "
        "A methodological conclusion is that small accuracy gaps from single splits are unreliable, and that "
        "repeated resampling with corrected significance testing should be standard in library and model comparisons. "
        "The accompanying code regenerates every table and figure in this paper.")

    P.h1("Declarations")
    P.p("Data and code availability: both datasets are distributed with Scikit-learn. Code: ", after=2)
    P.p("[INSERT GITHUB / ZENODO URL OF run_experiments.py AND build_paper.py BEFORE SUBMISSION]", highlight=True, bold=True, after=4)
    P.p("Use of AI tools, funding and conflict-of-interest statements: [COMPLETE ACCORDING TO THE TARGET JOURNAL’S POLICY]",
        highlight=True, bold=True)

    P.h1("References")
    for r in REFERENCES:
        para = P.p(r, align="left", after=3)
        para.paragraph_format.left_indent = Inches(0.35)
        para.paragraph_format.first_line_indent = Inches(-0.35)
    P.d.save(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="Research_Paper.docx")
    ap.add_argument("--fig", default="fig_accuracy.png")
    a = ap.parse_args()
    with open(a.results) as f:
        build(json.load(f), a.out, a.fig)
    print("saved", a.out)
