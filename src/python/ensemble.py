import numpy as np
import argparse

'''
python ensemble.py best/507_7_50_0.889_depth_l2_add_A3150_eval.txt best/best_3_0.889_l2_add_A3150_eval.txt best/7_50_0.888_depth_l2_add_A3150_eval.txt best/8_33_0.889_depth_l2_add_A3150_eval.txt best/8_35_0.888_depth_l2_add_A3150_eval.txt best/8_38_0.889_depth_l2_add_A3150_eval.txt best/5_9_0.889_depth_l2_add_A3150_eval.txt
'''
def adapt_labels(labels):
  max_label = np.max(labels) # 2 or 4

  if max_label == 2:
    labels += 2
  elif max_label == 4:
    labels -= 2
  else:
    print('Exception error! please check with np.unique(labels).')
  
  return labels

def get_args():
  parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')
  parser.add_argument('f1')
  parser.add_argument('f2')
  parser.add_argument('f3')
  parser.add_argument('f4')
  parser.add_argument('f5')
  parser.add_argument('f6')
  parser.add_argument('f7')

  return parser.parse_args()

def main():
  args = get_args()
  # 汚い
  # 任意個数の引数を渡せる関数を作成すべき
  res1 = np.loadtxt(args.f1)
  res2 = np.loadtxt(args.f2)
  res3 = np.loadtxt(args.f3)
  res4 = np.loadtxt(args.f4)
  res5 = np.loadtxt(args.f5)
  res6 = np.loadtxt(args.f6)
  res7 = np.loadtxt(args.f7)


  adapt_labels(res1)
  adapt_labels(res2)
  adapt_labels(res3)
  adapt_labels(res4)
  adapt_labels(res5)
  adapt_labels(res6)
  adapt_labels(res7)


  res1 = res1.astype('int64')
  res2 = res1.astype('int64')
  res3 = res1.astype('int64')
  res4 = res4.astype('int64')
  res5 = res5.astype('int64')
  res6 = res6.astype('int64')
  res7 = res7.astype('int64')


  # 多数決
  results = np.array([res1,res2,res3,res4,res5,res6,res7])
  ans = [*map(lambda x: np.argmax(np.bincount(x)), results.T)]
  ans = np.array(ans)
  adapt_labels(ans)
  
  with open('res/ensemble_eval.txt','w') as f:
    for y_pred in ans:
      y_pred = int(y_pred)
      y_pred = str(y_pred)
      f.write(y_pred + '\n')

if __name__ == '__main__':
    main()