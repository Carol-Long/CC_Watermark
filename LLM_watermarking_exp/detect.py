from watermark.old_watermark import OldWatermarkDetector
from watermark.cc_watermark import CCWatermarkDetector, K_CCWatermarkDetector
from watermark.our_watermark import NewWatermarkDetector
from watermark.gptwm import GPTWatermarkDetector
from watermark.watermark_v2 import WatermarkDetector
from watermark.linear_code import LinearCodeWatermarkDetector
from watermark.inverse_transform import InverseTransformDetector
from watermark.qarry_linear_code import Q_LinearCodeWatermarkDetector
from tqdm import tqdm
from pred import load_model_and_tokenizer, seed_everything, str2bool
import argparse
import os
import json
import torch
import pdb
from scipy.stats import norm
import re
import scipy 
from watermark.exponential import ExponentialWatermarkDetector
import numpy as np

def main(args):
    seed_everything(args.initial_seed_llm)
    model2path = json.load(open("config/model2path.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get model name
    model_name = args.input_dir.split("/")[-1].split("_")[0]
    # define your model
    # tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, load_token_only=True)
    tokenizer = load_model_and_tokenizer("meta-llama/Llama-2-7b-chat-hf", model_name, device, load_token_only=True)
    all_token_ids = list(tokenizer.get_vocab().values())
    vocab_size = len(all_token_ids)
    # get gamma and delta
    # pdb.set_trace()
    if "fresh" in args.input_dir:
        print(f'fresh seed generation')
        args.dynamic_seed = "fresh"
    if "gpt" in args.input_dir or 'old' in args.input_dir or 'no' in args.input_dir or 'v2' in args.input_dir:
        # pdb.set_trace()
        if 'poem' in args.input_dir and 'fresh' not in args.input_dir:
            delta = float(args.input_dir.split("_d")[1].split("/")[0])
        else:
            delta = float(args.input_dir.split("_d")[1].split("_")[0])
        gamma = float(args.input_dir.split("_g")[1].split("_")[0])
    else:
        gamma = delta = 0.0
    print(f"gamma is: {gamma}, delta is: {delta}")
    if 'q_lin_code' in args.input_dir:
        # q = int(args.input_dir.split("_q")[1].split("_")[0])
        # pdb.set_trace() 
        q = int(args.input_dir.rsplit("_q_", 1)[1].split("_", 1)[0])
        print(f'q is: {q}')
    
    
    # get all files from input_dir
    files = os.listdir(args.input_dir)
    # get all json files
    json_files = [f for f in files if f.endswith(".jsonl")]
    os.makedirs(args.input_dir + "/z_score", exist_ok=True)
    if args.mission != "all":
        json_files = [f for f in files if args.mission in f]
    for json_file in json_files:
        print(f"{json_file} has began.........")
        
        # for debugging on a single dataset prediction:
        # if json_file != "short_finance_qa.jsonl":
        #     continue
        
        # read jsons
        # pdb.set_trace()
        with open(os.path.join(args.input_dir, json_file), "r") as f:
            # lines
            lines = f.readlines()
            # texts
            for i, line in enumerate(lines):
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"\n[Error] Line {i}: {repr(line)}\n{e}")
                    print(f"output directory is: {args.input_dir + '/z_score'}")
                    break
            prompts = [json.loads(line)["prompt"] for line in lines]
            texts = [json.loads(line)["pred"] for line in lines]
            # print(f"texts[0] is: {texts[0]}")
            tokens = [json.loads(line)["completions_tokens"] for line in lines]
            
            
        
        if "old" in args.input_dir or "no" in args.input_dir:
            detector = OldWatermarkDetector(tokenizer=tokenizer,
                                            vocab=all_token_ids,
                                            gamma=gamma,
                                            delta=delta,
                                            dynamic_seed=args.dynamic_seed,
                                            device=device)
        
        if "new" in args.input_dir:
            detector = NewWatermarkDetector(tokenizer=tokenizer,
                                        vocab=all_token_ids,
                                        gamma=gamma,
                                        delta=delta,
                                        dynamic_seed=args.dynamic_seed,
                                        device=device,
                                        # vocabularys=vocabularys,
                                        )
            
        if "v2" in args.input_dir:
            detector = WatermarkDetector(
                vocab=all_token_ids,
                gamma=gamma,
                z_threshold=args.threshold,tokenizer=tokenizer,
                seeding_scheme=args.seeding_scheme,
                device=device,
                normalizers=args.normalizers,
                ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                select_green_tokens=args.select_green_tokens)
            
        if "gpt" in args.input_dir:
            detector = GPTWatermarkDetector(
                fraction=gamma,
                strength=delta,
                vocab_size=vocab_size,
                watermark_key=args.wm_key)

        if "cc-k" in args.input_dir:  
            # pdb.set_trace()
            print('performing detection fo cc')
            # Extract the number after "_k_" in args.input_dir
            match = re.search(r"_k_(\d+)", args.input_dir)
            if match:
                cc_k = int(match.group(1))  # Extracted number as an integer
                print(f"Extracted k value: {cc_k}")
            else:
                print("No '_k_' followed by a number found in input_dir.")
            detector = K_CCWatermarkDetector(tokenizer=tokenizer,
                                            vocab=all_token_ids,
                                            gamma=gamma,
                                            delta=delta,
                                            dynamic_seed=args.dynamic_seed,
                                            device=device,
                                            k=cc_k)
            z_s_score_list = []

        elif "cc" in args.input_dir or "cc-combined" in args.input_dir:  
            print('performing detection fo cc')
            detector = CCWatermarkDetector(tokenizer=tokenizer,
                                            vocab=all_token_ids,
                                            gamma=gamma,
                                            delta=delta,
                                            dynamic_seed=args.dynamic_seed,
                                            device=device)
            z_s_score_list = []
        
        
        elif "exponential" in args.input_dir:  
            print('performing detection for exponential')
            detector = ExponentialWatermarkDetector(tokenizer=tokenizer,
                                            vocab=all_token_ids,
                                            dynamic_seed=args.dynamic_seed,
                                            device=device)
            teststats_list = []
            gen_token_length_list = []
        
        elif "inv_tr" in args.input_dir:  
            print('performing detection for inverse transform')
            detector = InverseTransformDetector(tokenizer=tokenizer,
                                            vocab=all_token_ids,
                                            dynamic_seed=args.dynamic_seed,
                                            device=device)
            teststats_list = []
            p_vals_list = []

        # pdb.set_trace()
        z_score_list = []
        i=0
        for idx, cur_text in tqdm(enumerate(texts), total=len(texts)):
            #print("cur_text is:", cur_text)

            gen_tokens = tokenizer.encode(cur_text, return_tensors="pt", truncation=True, add_special_tokens=False)
            #print("gen_tokens is:", gen_tokens)
            prompt = prompts[idx]
            
            input_prompt = tokenizer.encode(prompt, return_tensors="pt", truncation=True,add_special_tokens=False)
            
        
            
            if len(gen_tokens[0]) >= args.test_min_tokens:

                if "v2" in args.input_dir:
                    z_score_list.append(detector.detect(cur_text)["z_score"])
                    
                elif "old" in args.input_dir or "no" in args.input_dir:
                    # print("gen_tokens is:", gen_tokens)
                    z_score_list.append(detector.detect(tokenized_text=gen_tokens, inputs=input_prompt))
                
                elif "cc" in args.input_dir:
                    # print("gen_tokens is:", gen_tokens)
                    z = detector.detect(tokenized_text=gen_tokens, inputs=input_prompt)
                    z_score_list.append(z)

                
                elif "inv_tr" in args.input_dir:
                    # print("gen_tokens is:", gen_tokens)
                    teststats,p_val,z = detector.detect(tokenized_text=gen_tokens, inputs=input_prompt)
                    z_score_list.append(z)
                    p_vals_list.append(p_val)
                    teststats_list.append(teststats)

                
                elif "exponential" in args.input_dir:
                    # print("gen_tokens is:", gen_tokens)
                    z_scores, teststats, gen_token_length = detector.detect(tokenized_text=gen_tokens, inputs=input_prompt)
                    z_score_list.append(z_scores)
                    teststats_list.append(teststats)
                    gen_token_length_list.append(gen_token_length)

                
                elif "gpt" in args.input_dir:
                      z_score_list.append(detector.detect(gen_tokens[0]))
                elif "new" in args.input_dir:
                      z_score_list.append(detector.detect(tokenized_text=gen_tokens, tokens=tokens[idx], inputs=input_prompt))
            
            
            else:   
                print(f"Warning: sequence {idx} is too short to test.")
            
            i += 1

        if 'exponential' in args.input_dir:
            #pdb.set_trace()
            # p_value for each 

            pval = [scipy.stats.gamma.sf(test_stats, T, scale = 1.0) for test_stats, T in zip(teststats_list, gen_token_length_list)]
            average_pval = torch.mean(torch.tensor(pval)).item()
            #thresholding
            pvalue = 0.001
            threshold_list = [scipy.stats.gamma.isf(pvalue, a = T, scale = 1.0) for T in gen_token_length_list]
            ##
            agg_pvals = [-np.log10(pv+1e-16) for pv in pval]
            ##
            save_dict = {
                'agg_pvals_avg': np.mean(agg_pvals),
                'agg_pvals_stder': np.std(agg_pvals)/np.sqrt(len(agg_pvals)),
                'pval_teststats': average_pval,
                'pval_std': torch.std(torch.tensor(pval)).item(),
                'wm_pred': [1 if z > threshold else 0 for z,threshold in zip(teststats_list, threshold_list)],
                'avarage_z': torch.mean(torch.tensor(z_score_list)).item(),
                'z_score_list': z_score_list,
                'z_std': torch.std(torch.tensor(z_score_list)).item(),
                'teststats_list': teststats_list,
                'gen_token_length_list': gen_token_length_list
            }
            z_file = json_file.replace('.jsonl', f'_{gamma}_{delta}_{args.threshold}_z.jsonl')
            output_path = os.path.join(args.input_dir + "/z_score", z_file)
            output_dir = args.input_dir + "/z_s_score"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_path, 'w') as fout:
                json.dump(save_dict, fout)
            continue

        if 'inv_tr' in args.input_dir:
            #pdb.set_trace()
            # p_value for each 

            average_pval = torch.mean(torch.tensor(p_vals_list)).item()
            #thresholding
            pvalue = 0.001
            # threshold_list = [scipy.stats.gamma.isf(pvalue, a = T, scale = 1.0) for T in gen_token_length_list]
            ##
            agg_pvals = [-np.log10(pv+1e-16) for pv in p_vals_list]
            ##
            save_dict = {
                'agg_pvals_avg': np.mean(agg_pvals),
                'agg_pvals_stder': np.std(agg_pvals)/np.sqrt(len(agg_pvals)),
                'pval_teststats': average_pval,
                'pval_std': torch.std(torch.tensor(p_vals_list)).item(),
                # 'wm_pred': [1 if z > threshold else 0 for z,threshold in zip(teststats_list, threshold_list)],
                # 'avarage_z': torch.mean(torch.tensor(z_score_list)).item(),
                # 'z_score_list': z_score_list,
                # 'std_z': torch.std(torch.tensor(z_s_score_list)).item(),
                'teststats_list': teststats_list,
                # 'gen_token_length_list': gen_token_length_list
            }
            z_file = json_file.replace('.jsonl', f'_{gamma}_{delta}_{args.threshold}_z.jsonl')
            output_path = os.path.join(args.input_dir + "/z_score", z_file)
            output_dir = args.input_dir + "/z_s_score"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_path, 'w') as fout:
                json.dump(save_dict, fout)
            continue
        

        else:
            p_val = 1 - norm.cdf(torch.mean(torch.tensor(z_score_list)).item())
            p_vals = [1-norm.cdf(z_) for z_ in z_score_list]
            ##
            agg_pvals = [-np.log10(pv+1e-16) for pv in p_vals]
            ##
            save_dict = {
                'agg_pvals_avg': np.mean(agg_pvals),
                'agg_pvals_stder': np.std(agg_pvals)/np.sqrt(len(agg_pvals)),
                'p_val_teststats': p_vals,
                'avg_pval': torch.mean(torch.tensor(p_vals)).item(),
                'std_pval': torch.std(torch.tensor(p_vals)).item(),
                'z_score_list': z_score_list,
                'avarage_z': torch.mean(torch.tensor(z_score_list)).item(),
                'std_z': torch.std(torch.tensor(z_score_list)).item(),
                'wm_pred': [1 if z > args.threshold else 0 for z in z_score_list]
                }
            
            wm_pred_average = torch.mean(torch.tensor(save_dict['wm_pred'], dtype=torch.float))
            save_dict.update({'wm_pred_average': wm_pred_average.item()})   
            
            print(save_dict)
            # average_z = torch.mean(z_score_list)
            z_file = json_file.replace('.jsonl', f'_{gamma}_{delta}_{args.threshold}_z.jsonl')
            output_path = os.path.join(args.input_dir + "/z_score", z_file)
            with open(output_path, 'w') as fout:
                json.dump(save_dict, fout)

            
        if "cc" in args.input_dir:
            p_val = 1 - norm.cdf(torch.mean(torch.tensor(z_score_list)).item())
            # pdb.set_trace()
            p_vals = [1-norm.cdf(z_) for z_ in z_score_list]
            ##
            agg_pvals = [-np.log10(pv+1e-16) for pv in p_vals]
            ##
            save_dict = {
            'agg_pvals_avg': np.mean(agg_pvals),
            'agg_pvals_stder': np.std(agg_pvals)/np.sqrt(len(agg_pvals)),
            'p_val_teststats': p_vals,
            'avg_pval': torch.mean(torch.tensor(p_vals)).item(),
            'std_pval': torch.std(torch.tensor(p_vals)).item(),
            'z_score_list': z_s_score_list,
            'avarage_z': torch.mean(torch.tensor(z_s_score_list)).item(),
            'std_z': torch.std(torch.tensor(z_s_score_list)).item(),
            'wm_pred': [1 if z > args.threshold else 0 for z in z_s_score_list]
            }
            
            wm_pred_average = torch.mean(torch.tensor(save_dict['wm_pred'], dtype=torch.float))
            save_dict.update({'wm_pred_average': wm_pred_average.item()})   
            
            # print(save_dict)
            # average_z = torch.mean(z_score_list)
            z_file = json_file.replace('.jsonl', f'_{gamma}_{delta}_{args.threshold}_z.jsonl')
            output_path = os.path.join(args.input_dir + "/z_s_score", z_file)
            output_dir = args.input_dir + "/z_s_score"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_path, 'w') as fout:
                json.dump(save_dict, fout)
            
            



parser = argparse.ArgumentParser(description="Process watermark to calculate z-score for every method")

parser.add_argument(
    "--input_dir",
    type=str,
    default="pred/llama2-7b-chat-4k_cc_g0.5_d5.0")
parser.add_argument( # for gpt watermark
        "--wm_key", 
        type=int, 
        default=0)

parser.add_argument(
    "--threshold",
    type=float,
    default=3.05)

parser.add_argument(
    "--test_min_tokens",
    type=int, 
    default=2)

parser.add_argument( # for v2 watermark
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )

parser.add_argument( # for v2 watermark
    "--normalizers",
    type=str,
    default="",
    help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
)

parser.add_argument( # for v2 watermark
    "--ignore_repeated_bigrams",
    type=str2bool,
    default=False,
    help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
)

parser.add_argument( # for v2 watermark
    "--select_green_tokens",
    type=str2bool,
    default=True,
    help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
)

parser.add_argument( 
    "--mission",
    type=str,
    default="all",
    help="mission-name",
)

parser.add_argument(
    "--initial_seed_llm",
    type=int,
    default=42)

parser.add_argument(
        "--dynamic_seed",
        type=str,
        default="markov_1",
        choices=[None, "initial", "markov_1","fresh"],
        help="The seeding procedure to use when sampling the redlist at each step.",
        )
args = parser.parse_args()

main(args)

