import sys
import re
import math
import random 
from collections import defaultdict, Counter

#------------------------text preprocessing-------------------
def pre_processing(filename):
    instances = []
    with open(filename, 'r') as file:
        content = file.read()
    
    # Getting target word from file name (assuming the target word is the filename by default)
    ambiguity_word = filename.split('.')[0]  
    for i in re.finditer(r'<instance id="(.*?)".*?<answer .*? senseid="(.*?)".*?<context>(.*?)</context>', content, re.DOTALL):
        instance_id = i.group(1)
        sense_id = i.group(2)
        context = i.group(3)
        
        # Removing target word and <head> tags from context
        context = re.sub(rf'<head>{ambiguity_word}</head>', '', context, flags=re.IGNORECASE)
        words = re.findall(r'\b\w+\b', context.lower())
        
        instances.append((instance_id, sense_id, words))
    #for instance in instances[:3]:  
        #print(f"ID: {instance[0]}, Sense: {instance[1]}, Context words: {instance[2]}")
    
    return instances

#---------------------------------folds---------------------------

def helper_folds(instances, num_folds=5, shuffle_data=False):
    if shuffle_data:
        random.shuffle(instances)  

    n = len(instances)
    fold_size = math.ceil(n / num_folds)  
    folds = []
    starting = 0
    for i in range(num_folds - 1):
        ending = starting + fold_size
        folds.append(instances[starting:ending])
        starting = ending
    folds.append(instances[starting:])

    #print(f"Folds are {num_folds} folds with sizes: {[len(i) for i in folds]}")
    #print(folds[0])
    return folds

#--------------------counts-----------------------

def counts_naive_bayes(train_set):
    w_c = defaultdict(lambda: defaultdict(int))
    s_c = Counter()
    total_w = defaultdict(int)
    vocab = set()
    
    for _, sense, words in train_set:
        s_c[sense] += 1
        for word in words:
            w_c[sense][word] += 1
            total_w[sense] += 1
            vocab.add(word)
    
    return w_c, s_c, total_w, vocab

#------------Naive Bayes--------------------------

def Naive_bayes(words, w_c, s_c, total_w, vocab):
    probs = {}
    V = len(vocab)
    
    for sense in s_c:
        # Calculating log P(sense)
        log_prob = math.log(s_c[sense] / sum(s_c.values()))
        
        for i in words:
            # Calculating log P(word|sense) with add-one smoothing
            word_prob = (w_c[sense][i] + 1) / (total_w[sense] + V)
            log_prob += math.log(word_prob)
        
        probs[sense] = log_prob
    
    # Returning the sense with the maximum probability
    return max(probs, key=probs.get)

#-----------------------Testing----------------

def testing(instances, num_folds=5, shuffle_data=False):
    folds = helper_folds(instances, num_folds, shuffle_data)
    f_accuracies = []
    output_lines = []
    misclassified_instances = []  
    
    for i in range(num_folds):
        test_set = folds[i]
        train_set = [instance for j, fold in enumerate(folds) if j != i for instance in fold]
      
        w_c, s_c, total_w, vocab = counts_naive_bayes(train_set)
 
        correct = 0
        each_fold_output = [f"Fold {i+1}"]
        
        for i, true_sense, words in test_set:
            predicted_sense = Naive_bayes(words, w_c, s_c, total_w, vocab)
            each_fold_output.append(f"{i} {predicted_sense}")
            
            if predicted_sense == true_sense:
                correct += 1
            #ignore the mispaced instances .it is just for my analysis
            else:
                misclassified_instances.append((i, true_sense, predicted_sense))
            
        accuracy = correct / len(test_set) * 100
        f_accuracies.append(accuracy)
        
        output_lines.extend(each_fold_output)
    
   
    print("Fold Accuracies are:")
    for i, acc in enumerate(f_accuracies, 1):
        print(f"Fold {i}: {acc:.2f}%")
    print(f"Average Accuracy: {sum(f_accuracies) / len(f_accuracies):.2f}%")


    #print("\nMisclassified Instances are:")
    #for i, j, k in misclassified_instances:
        #print(f"Instance_ID: {i}, True_Sense: {j}, Predicted_Sense: {k}")
        

    out_file = sys.argv[1].replace(".wsd", ".wsd.out")
    with open(out_file, 'w') as f:
        f.write("\n".join(output_lines))
    
    return f_accuracies

#----------------main-------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please use this command - python wsd.py <input_filename>")
        sys.exit("Error - Missing input file or check the name of the Python file.")
    
    input_file = sys.argv[1]
    instances = pre_processing(input_file)
    testing(instances)
