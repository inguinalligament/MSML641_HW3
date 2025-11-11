################################################################################################
####        TITLE: MSML HW3                                                                 ####
####        DESCRIPTION: SENTIMENT ANALYSIS - TRAIN.PY                                      ####
####        AUTHOR: BRADLEY SCOTT                                                           ####
####        UMD ID: 119 775 028                                                             ####
####        DATE: 26OCT2025                                                                 ####
####        REFERENCES USED (see paper for full details):                                   ####
####            ChatGPT 5                                                                   ####
################################################################################################

'''
[BS10262025] p3_641_000001
[BS10262025] install any non standard modules
'''
import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
        print(f"✅ {package} is already installed.")
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_if_missing("tensorflow")
install_if_missing("pandas")
install_if_missing("scikit-learn")
install_if_missing("matplotlib")


'''
[BS10262025] p3_641_000005
[BS10262025] import all necessary modules
'''
import argparse, os
from preprocess import load_and_preprocess, pad_data
from models import build_model
from evaluate import evaluate_model
from utils import set_seed, timer, save_results_to_csv

'''
[BS10262025] p3_641_000010
[BS10262025] Set the seed number
'''
set_seed(42)

'''
[BS10272025] p3_641_000011
[BS10272025] Build a function that will run one iteration of the model
'''
def run_one(arch, act, opt, L, stab, Xtr, Xte, y_train, y_test, args, results):
    grad_clip_flag = "Yes" if stab != "none" else "No"

    model = build_model(
        arch=arch,
        vocab_size=args.vocab_size,
        seq_len=L,
        optimizer_name=opt,
        activation=act,
        stability=stab
    )
    (_, train_time) = timer(model.fit)(
        Xtr, y_train,
        batch_size=args.batch,
        epochs=args.epochs,
        validation_split=0.1,
        verbose=2
    )
    acc, f1 = evaluate_model(model, Xte, y_test)
    results.append({
        "Model": "Bidirectional LSTM" if arch == "bilstm" else arch.upper(),
        "Activation": act.capitalize(),
        "Optimizer": opt.lower(),
        "Seq Length": L,
        "Grad Clipping": grad_clip_flag,
        "Accuracy": acc,
        "F1": f1,
        "Epoch Time (s)": train_time / args.epochs
    })
    print(f"Acc={acc:.4f}, F1={f1:.4f}, Epoch Time={train_time/args.epochs:.2f}s")
    return acc, f1

'''
[BS10262025] p3_641_000020
[BS10262025] Build out all the arguments for the main function and the calls to preprocess, model, evaluate and utils
            NB: This goes through a one factor at a time update to the model where it chooses the best
                option from the current choices(mode, activation, optimizer, sequence length, etc...)
                while the rest of the variables are held steady. 
                The alternative of a full grid search had a theoretical run time of over 10 hours.
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--results", type=str, default="./results/metrics.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)

    # Load data (tokenize on train)
    X_train, X_test, y_train, y_test, _ = load_and_preprocess(args.data, vocab_size=args.vocab_size)

    # Baseline configuration
    base_arch = "lstm"
    base_act  = "tanh"
    base_opt  = "adam"
    base_len  = 50
    base_stab = "none"

    def pad_for(L):
        return pad_data(X_train, X_test, L)

    results = []

    # Stage 1: Architectures (hold others fixed)
    print("\n[Stage 1] Architectures (rnn, lstm, bilstm) with baseline settings")
    Xtr, Xte = pad_for(base_len)
    best = {"acc": -1, "arch": None, "act": base_act, "opt": base_opt, "len": base_len, "stab": base_stab}
    for arch in ["rnn", "lstm", "bilstm"]:
        print(f"\n-- Testing architecture: {arch}")
        acc, _ = run_one(arch, base_act, base_opt, base_len, base_stab, Xtr, Xte, y_train, y_test, args, results)
        if acc > best["acc"]:
            best.update(acc=acc, arch=arch)

    # Stage 2: Activations (with best arch)
    print("\n[Stage 2] Activations (tanh, relu, sigmoid) with best architecture")
    for act in ["tanh", "relu", "sigmoid"]:
        print(f"\n-- Testing activation: {act}")
        acc, _ = run_one(best["arch"], act, best["opt"], best["len"], best["stab"], Xtr, Xte, y_train, y_test, args, results)
        if acc > best["acc"]:
            best.update(acc=acc, act=act)

    # Stage 3: Optimizers (with best arch+activation)
    print("\n[Stage 3] Optimizers (adam, sgd, rmsprop) with best arch+activation")
    for opt in ["adam", "sgd", "rmsprop"]:
        print(f"\n-- Testing optimizer: {opt}")
        acc, _ = run_one(best["arch"], best["act"], opt, best["len"], best["stab"], Xtr, Xte, y_train, y_test, args, results)
        if acc > best["acc"]:
            best.update(acc=acc, opt=opt)

    # Stage 4: Sequence lengths (25, 50, 100) with best arch+act+opt
    print("\n[Stage 4] Sequence length (25, 50, 100) with best arch+act+opt")
    for L in [25, 50, 100]:
        print(f"\n-- Testing seq length: {L}")
        Xtr, Xte = pad_for(L)
        acc, _ = run_one(best["arch"], best["act"], best["opt"], L, best["stab"], Xtr, Xte, y_train, y_test, args, results)
        if acc > best["acc"]:
            best.update(acc=acc, len=L)

    # Stage 5: Stability (no strategy vs clipnorm) with best arch+act+opt+len
    print("\n[Stage 5] Stability (none vs clipnorm) with best arch+act+opt+len")
    Xtr, Xte = pad_for(best["len"])
    for stab in ["none", "clipnorm"]:
        print(f"\n-- Testing stability: {stab}")
        acc, _ = run_one(best["arch"], best["act"], best["opt"], best["len"], stab, Xtr, Xte, y_train, y_test, args, results)
        if acc > best["acc"]:
            best.update(acc=acc, stab=stab)

    # Final confirmation run using best config (repeat once more for clarity)
    print("\n[Final] Confirmation run with best configuration")
    Xtr, Xte = pad_for(best["len"])
    run_one(best["arch"], best["act"], best["opt"], best["len"], best["stab"], Xtr, Xte, y_train, y_test, args, results)

    save_results_to_csv(results, args.results)
    print("\nControlled design complete.\nBest configuration:")
    print(best)
    print("\nResults saved to:", args.results)

'''
[BS10262025] p3_641_000020
[BS10262025] Initiate the function
'''
if __name__ == "__main__":
    main()