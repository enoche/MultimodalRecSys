# Multimodal Recommender Systems
A curated list of awesome multimodal recommendation resources (***code first***: _messy code is better than no code_). Please contact us if your work is not in the list.  
*- Last updated: 2023/08/22 (MM'23 Partially Updated)*  
- :question: 作者未回应提供源代码请求。如大家有相关源代码或可以复现，请联系我。  
- :x: 不可复现  
- :heart: 希望大家一起把 [多模态推荐](https://github.com/enoche/MultimodalRecSys) 这个领域做扎实！  


```
@article{zhou2023comprehensive,
      title={A Comprehensive Survey on Multimodal Recommender Systems: Taxonomy, Evaluation, and Future Directions}, 
      author={Hongyu Zhou and Xin Zhou and Zhiwei Zeng and Lingzi Zhang and Zhiqi Shen},
      year={2023},
      journal={arXiv preprint arXiv:2302.04473},
}
```

## [A Unified Multimodal Recommendation Framework](https://github.com/enoche/MMRec), $\text{MMRec}$, which currently integrates:
- 10+ multimodal models
- Various general CF models


## Survey

- [A Comprehensive Survey on Multimodal Recommender Systems: Taxonomy, Evaluation, and Future Directions](https://arxiv.org/abs/2302.04473) - Hongyu Zhou, Xin Zhou, Zhiwei Zeng, Lingzi Zhang, Zhiqi Shen. **arxiv, Feb 2023**  
- [Recommender Systems Leveraging Multimedia Content](https://dl.acm.org/doi/10.1145/3407190) - Yashar Deldjoo, Markus Schedl, Paolo Cremonesi, Gabriella Pasi. **ACM Computing Surveys, Sep 2020**  


## Table of Contents
- [General](#general)
- [Modality-specific Recommendation](#modality-specific-recommendation)
  - [Text-based Recommendation](#text-based-recommendation)
  - [Image-based Recommendation](#image-based-recommendation)
- [Public Datasets](#public-datasets-link-verified-heavy_check_mark)

## General

- [Multi-View Graph Convolutional Network for Multimedia Recommendation](https://arxiv.org/abs/2308.03588) - Penghang Yu, Zhiyi Tan, Guanming Lu, Bing-Kun Bao. **arXiv, Aug 2023** | [`[pdf]`](https://arxiv.org/ftp/arxiv/papers/2308/2308.03588.pdf)  | [`[code]`](https://github.com/demonph10/MGCN) 
- [Pareto Invariant Representation Learning for Multimedia Recommendation](https://arxiv.org/abs/2308.04706) - Shanshan Huang, Haoxuan Li, Qingsong Li, Chunyuan Zheng, Li Liu. **arXiv, Aug 2023** | [`[pdf]`](https://arxiv.org/pdf/2308.04706.pdf)  
- [Bootstrap Latent Representations for Multi-modal Recommendation](https://arxiv.org/abs/2207.05969) - Xin Zhou, Hongyu Zhou, Yong Liu, Zhiwei Zeng, Chunyan Miao, Pengwei Wang, Yuan You, Feijun Jiang. **WWW, May 2023** | [`[pdf]`](https://arxiv.org/pdf/2207.05969.pdf) | [`[code]`](https://github.com/enoche/BM3)  
- [Multi-Modal Self-Supervised Learning for Recommendation](https://arxiv.org/abs/2302.10632) - Wei Wei, Chao Huang, Lianghao Xia, Chuxu Zhang. **WWW, May 2023** | [`[pdf]`](https://arxiv.org/pdf/2302.10632.pdf) | [`[code]`](https://github.com/HKUDS/MMSSL)  
- [Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation](https://arxiv.org/abs/2301.12097) - Hongyu Zhou, Xin Zhou, Lingzi Zhang, Zhiqi Shen. **arxiv, Feb 2023** | [`[pdf]`](https://arxiv.org/pdf/2301.12097.pdf) | [`[code]`](https://github.com/hongyurain/DRAGON)
- [A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation](https://arxiv.org/abs/2211.06924) - Xin Zhou. **arXiv, Nov 2022** | [`[pdf]`](https://arxiv.org/pdf/2211.06924.pdf) | [`[code]`](https://github.com/enoche/FREEDOM) 
- [Self-supervised Learning for Multimedia Recommendation](https://ieeexplore.ieee.org/document/9811387) - Zhulin Tao, Xiaohao Liu, Yewei Xia, Xiang Wang, Lifang Yang, Xianglin Huang, Tat-Seng Chua. **TMM, Jun 2022** | [`[code]`](https://github.com/zltao/SLMRec)
- [Latent Structure Mining with Contrastive Modality Fusion for Multimedia Recommendation](https://arxiv.org/abs/2111.00678) - Jinghao Zhang, Yanqiao Zhu, Qiang Liu, Mengqi Zhang, Shu Wu, Liang Wang. **arxiv, Mar 2022** (Extended version of LATTICE which has been published in MM21) | [`[pdf]`](https://arxiv.org/pdf/2111.00678.pdf)  | [`[code]`](https://github.com/cripac-dig/micro)
- [DualGNN: Dual Graph Neural Network for Multimedia Recommendation](https://ieeexplore.ieee.org/abstract/document/9662655)  - Qifan Wang, Yinwei Wei, Jianhua Yin, Jianlong Wu, Xuemeng Song, Liqiang Nie.  **IEEE TMM, Dec 2021**  | [`[code]`](https://github.com/wqf321/dualgnn)
- [Mining Latent Structures for Multimedia Recommendation](https://dl.acm.org/doi/pdf/10.1145/3474085.3475259) - Jinghao Zhang, Yanqiao Zhu, Qiang Liu, Shu Wu, Shuhui Wang, Liang Wang. **MM, Oct 2021**  | [`[pdf]`](https://arxiv.org/pdf/2104.09036.pdf)  | [`[code]`](https://github.com/CRIPAC-DIG/LATTICE)
- [Pre-training Graph Transformer with Multimodal Side Information for Recommendation](https://dl.acm.org/doi/abs/10.1145/3474085.3475709) - Yong Liu, Susen Yang, Chenyi Lei, Guoxin Wang, Haihong Tang, Juyong Zhang, Aixin Sun and Chunyan Miao. **MM, Oct 2021** | [`[pdf]`](https://arxiv.org/pdf/2010.12284.pdf) | [`[code]`](https://github.com/uoo723/PMGT)
- [Why Do We Click: Visual Impression-aware News Recommendation](https://dl.acm.org/doi/10.1145/3474085.3475514) - Jiahao Xun, Shengyu Zhang, Zhou Zhao, Jieming Zhu, Qi Zhang, Jingjie Li, Xiuqiang He, Xiaofei He, Tat-Seng Chua, Fei Wu. **MM, Oct 2021**  | [`[pdf]`](https://arxiv.org/pdf/2109.12651.pdf) | [`[code]`](https://github.com/JiahaoXun/IMRec)
- [Multi-Modal Variational Graph Auto-Encoder for Recommendation Systems](https://ieeexplore.ieee.org/document/9535249) - Jing Yi and Zhenzhong Chen. **IEEE TMM, Sep 2021**  | [`[code]`](https://github.com/jing-1/MVGAE) :x:
- [Recommendation by Users' Multimodal Preferences for Smart City Applications](https://ieeexplore.ieee.org/document/9152003) - Cai Xu , Ziyu Guan , Member, IEEE, Wei Zhao , Quanzhou Wu , Meng Yan, Long Chen and Qiguang Miao. **IEEE TII, Jul 2021**  | [`[code]`](https://github.com/winterant/UMPR) 
- [Adversarial Training Towards Robust Multimedia Recommender System](https://ieeexplore.ieee.org/document/8618394) - Jinhui Tang, Xiaoyu Du, Xiangnan He, Fajie Yuan, Qi Tian and Tat-Seng Chua. **IEEE TKDE, May 2020** | [`[pdf]`](https://arxiv.org/pdf/1809.07062.pdf) | [`[code]`](https://github.com/duxy-me/AMR)
- [MGAT: Multimodal Graph Attention Network for Recommendation](https://www.sciencedirect.com/science/article/abs/pii/S0306457320300182) - Zhulin Tao, Yinwei Wei, Xiang Wang, Xiangnan He, Xianglin Huang, Tat-Seng Chua. **Information Processing & Management, Apr 2020** | [`[code]`](https://github.com/zltao/MGAT)
- [Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback](https://dl.acm.org/doi/10.1145/3394171.3413556) - Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, Tat-Seng Chua. **MM, Oct 2020** [`[pdf]`](https://arxiv.org/pdf/2111.02036.pdf) | [`[code]`](https://github.com/weiyinwei/GRCN)
- [User Diverse Preference Modeling by Multimodal Attentive Metric Learning](https://dl.acm.org/doi/10.1145/3343031.3350953) - Fan Liu, Zhiyong Cheng, Changchang Sun, Yinglong Wang, Liqiang Nie, Mohan Kankanhalli. **MM, Oct 2019** | [`[pdf]`](https://arxiv.org/pdf/1908.07738.pdf) | [`[code]`](https://github.com/liufancs/MAML)
- [MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video](https://dl.acm.org/doi/10.1145/3343031.3351034) - Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, Richang Hong, Tat-Seng Chua. **MM, Oct 2019** | [`[pdf]`](http://staff.ustc.edu.cn/~hexn/papers/mm19-MMGCN.pdf) | [`[code]`](https://github.com/weiyinwei/MMGCN)

- [Breaking Isolation: Multimodal Graph Fusion for Multimedia Recommendation by Edge-wise Modulation](https://dl.acm.org/doi/abs/10.1145/3503161.3548399)  - Feiyu Chen, Junjie Wang, Yinwei Wei, Hai-Tao Zheng, Jie Shao. **MM, Oct 2022**
- [Learning Hybrid Behavior Patterns for Multimedia Recommendation](https://dl.acm.org/doi/abs/10.1145/3503161.3548119) - Zongshen Mu, Yueting Zhuang, Jie Tan, Jun Xiao, Siliang Tang. **MM, Oct 2022** :question: 
- [Multi-modal Graph Contrastive Learning for Micro-video Recommendation](https://dl.acm.org/doi/abs/10.1145/3477495.3532027) -  Zixuan Yi, Xi Wang, Iadh Ounis, Craig Macdonald. **SIGIR, Jul 2022** :question: 
- [MM-Rec: Visiolinguistic Model Empowered Multimodal News Recommendation](https://dl.acm.org/doi/abs/10.1145/3477495.3531896) - Chuhan Wu, Fangzhao Wu, Tao Qi, Chao Zhang, Yongfeng Huang, Tong Xu. **SIGIR, Jul 2022**
- [Disentangled Multimodal Representation Learning for Recommendation](https://arxiv.org/abs/2203.05406) - Fan Liu, Zhiyong Cheng, Huilin Chen, Anan Liu, Liqiang Nie, Mohan Kankanhalli. **arxiv, Mar 2022**  | [`[pdf]`](https://arxiv.org/pdf/2203.05406.pdf)
- [A two-stage embedding model for recommendation with multimodal auxiliary information](https://www.sciencedirect.com/science/article/abs/pii/S0020025521009270) - Juan Ni, Zhenhua Huang, Yang Hu, Chen Lin. **Information Sciences, Jan 2022**
- [Multi-modal Knowledge Graphs for Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3340531.3411947) - Rui Sun, Xuezhi Cao, Yan Zhao, Junchen Wan, Kun Zhou, Fuzheng Zhang, Zhongyuan Wang and Kai Zheng. **CIKM, Oct 2020** [`[pdf]`](https://zheng-kai.com/paper/cikm_2020_sun.pdf)
- [User-Video Co-Attention Network for Personalized Micro-video Recommendation](https://dl.acm.org/doi/10.1145/3308558.3313513) -Shang Liu, Zhenzhong Chen, Hongyi Liu, Xinghai Hu. **WWW, May 2019**
- [Personalized Fashion Recommendation with Visual Explanations based on Multimodal Attention Network](https://dl.acm.org/doi/10.1145/3331184.3331254) - Xu Chen, Hanxiong Chen, Hongteng Xu, Yongfeng Zhang, Yixin Cao, Zheng Qin, Hongyuan Zha. **SIGIR, Jul 2019**
- [Multimodal Representation Learning for Recommendation in Internet of Things](https://ieeexplore.ieee.org/document/8832204) - Zhenhua Huang, Xin Xu, Juan Ni, Honghao Zhu, and Cheng Wang. **IEEE IoTJ, Sep 2019**
- [GraphCAR: Content-aware Multimedia Recommendation with Graph Autoencoder](https://dl.acm.org/doi/10.1145/3209978.3210117) - Qidi Xu, Fumin Shen, Li Liu, Heng Tao Shen. **SIGIR, Jun 2018**

## Modality-specific Recommendation
### Text-based Recommendation

#### Review-based ####

- [Counterfactual Review-based Recommendation](https://dl.acm.org/doi/abs/10.1145/3459637.3482244) - Kun Xiong, Wenwen Ye, Xu Chen, Yongfeng Zhang, Wayne Xin Zhao, Binbin Hu, Zhiqiang Zhang, Jun Zhou. **CIKM, Nov 2021** | [`[pdf]`](https://librahu.github.io/data/cikm2021-CFCF-paper.pdf) | [`[code]`](https://github.com/CFCF-anonymous/Counterfactual-Review-based-Recommendation)
- [Reviews Meet Graphs: Enhancing User and Item Representations for Recommendation with Hierarchical Attentive Graph Neural Network](https://aclanthology.org/D19-1494/) - Chuhan Wu, Fangzhao Wu, Tao Qi, Suyu Ge, Yongfeng Huang, Xing Xie. **EMNLP-IJCNLP, Nov 2019** | [`[pdf]`](https://aclanthology.org/D19-1494.pdf) | [`[code]`](https://github.com/wuch15/Reviews-Meet-Graphs)
- [NRPA: Neural Recommendation with Personalized Atention](https://dl.acm.org/doi/10.1145/3331184.3331371) - Hongtao Liu, Fangzhao Wu, Wenjun Wang, Xianchen Wang, Pengfei Jiao, Chuhan Wu, Xing Xie. **SIGIR, Jul 2019** | [`[pdf]`](https://arxiv.org/pdf/1905.12480.pdf) | [`[code]`](https://github.com/microsoft/recommenders)
- [Recommendation Based on Review Texts and Social Communities: A Hybrid Model](https://ieeexplore.ieee.org/document/8635542) - Zhenyan Ji, Huaiyu Pi, Wei Wei, Bo Xiong, Marcin Woźniak, Robertas Damasevicius. **IEEE Access, Feb 2019** | [](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8635542) | [`[code]`](https://github.com/pp1230/HybridRecommendation)
- [A Context-Aware User-Item Representation Learning for Item Recommendation](https://dl.acm.org/doi/fullHtml/10.1145/3298988) -  Libing Wu, Cong Quan, Chenliang Li, Qian Wang, Bolong Zheng, Xiangyang Luo. **TOIS, Jan 2019** | [`[pdf]`](https://arxiv.org/pdf/1712.02342.pdf) | [`[code]`](https://github.com/WHUIR/CARL)
- [ANR: Aspect-based Neural Recommender](https://dl.acm.org/doi/10.1145/3269206.3271810) - Jin Yao Chin, Kaiqi Zhao, Shafiq Joty, Gao Cong. **CIKM, Oct 2018** | [`[pdf]`](https://kaiqizhao.github.io/cikm18-anr.pdf) | [`[code]`](https://github.com/almightyGOSU/ANR)
- [PARL: Let Strangers Speak Out What You Like](https://dl.acm.org/doi/10.1145/3269206.3271695) - Libing Wu, Cong Quan, Chenliang Li, Donghong Ji. **CIKM, Oct 2018** | [`[code]`](https://github.com/WHUIR/PARL)
- [Multi-Pointer Co-Attention Networks for Recommendation](https://dl.acm.org/doi/10.1145/3219819.3220086) - Yi Tay, Anh Tuan Luu, Siu Cheung Hui. **KDD, Jul 2018** | [`[pdf]`](https://arxiv.org/pdf/1801.09251.pdf) | [`[code]`](https://github.com/vanzytay/KDD2018_MPCN)
- [Neural Attentional Rating Regression with Review-level Explanations](https://dl.acm.org/doi/10.1145/3178876.3186070) - Chong Chen, Min Zhang, Yiqun Liu, Shaoping Ma. **WWW, Apr 2018** | [`[pdf]`](https://dl.acm.org/doi/pdf/10.1145/3178876.3186070) | [`[code]`](https://github.com/chenchongthu/NARRE)
- [Interpretable Convolutional Neural Networks with Dual Local and Global Attention for Review Rating Prediction](https://dl.acm.org/doi/10.1145/3109859.3109890) - Sungyong Seo, Jing Huang, Hao Yang, Yan Liu. **RecSys, Aug 2017** | [`[pdf]`](https://dl.acm.org/doi/pdf/10.1145/3109859.3109890) | [`[code]`](https://github.com/seongjunyun/CNN-with-Dual-Local-and-Global-Attention)
- [Joint Deep Modeling of Users and Items Using Reviews for Recommendation](https://dl.acm.org/doi/10.1145/3018661.3018665) - Lei Zheng, Vahid Noroozi, Philip S. Yu. **WSDM, Feb 2017** | [`[pdf]`](https://dl.acm.org/doi/pdf/10.1145/3018661.3018665) | [`[code]`](https://github.com/winterant/DeepCoNN)
- [Convolutional Matrix Factorization for Document Context-Aware Recommendation](https://dl.acm.org/doi/10.1145/2959100.2959165) - Donghyun Kim, Chanyoung Park, Jinoh Oh, Sungyoung Lee, Hwanjo Yu. **RecSys, Sep 2016** | [`[pdf]`](http://uclab.khu.ac.kr/resources/publication/C_351.pdf) | [`[code]`](https://github.com/cartopy/ConvMF)


- [Review-Aware Neural Recommendation with Cross-Modality Mutual Attention](https://dl.acm.org/doi/abs/10.1145/3459637.3482172) - Songyin Luo, Xiangkui Lu, Jun Wu, Jianbo Yuan. **CIKM, Oct. 2021**
- [Learning Hierarchical Review Graph Representations for Recommendation](https://ieeexplore.ieee.org/document/9416173) - Yong Liu, Susen Yang, Yinan Zhang, Chunyan Miao, Zaiqing Nie, Juyong Zhang. **IEEE TKDE, Apr 2021** | [`[pdf]`](https://arxiv.org/pdf/2004.11588.pdf)
- [Improving Explainable Recommendations by Deep Review-Based Explanations](https://ieeexplore.ieee.org/document/9417205) - Sixun Ouyang, Aonghus Lawlor. **IEEE Access, Apr 2021** | [`[pdf]`](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9417205)

- [Neural Unified Review Recommendation with Cross Attention](https://dl.acm.org/doi/abs/10.1145/3397271.3401249) - Hongtao Liu, Wenjun Wang, Hongyan Xu, Qiyao Peng, Pengfei Jiao. **SIGIR, Jul 2020**
- [DAML: Dual Attention Mutual Learning between Ratings and Reviews for Item Recommendation](https://dl.acm.org/doi/10.1145/3292500.3330906) - Donghua Liu, Jing Li, Bo Du, Jun Chang, Rong Gao. **KDD, Aug 2019**
- [Attentive Aspect Modeling for Review-Aware Recommendation](https://dl.acm.org/doi/10.1145/3309546) -  Xinyu Guan, Zhiyong Cheng, Xiangnan He, Yongfeng Zhang, Zhibo Zhu, Qinke Peng, Tat-Seng Chua. **TOIS, Mar 2019** | [`[pdf]`](https://arxiv.org/pdf/1811.04375.pdf)
- [Coevolutionary Recommendation Model: Mutual Learning between Ratings and Reviews](https://dl.acm.org/doi/10.1145/3178876.3186158) -Yichao Lu, Ruihai Dong, Barry Smyth. **WWW, Apr 2018** | [`[pdf]`](https://dl.acm.org/doi/pdf/10.1145/3178876.3186158)

#### Title, abstract, tag ####

- [Aligning Dual Disentangled User Representations from Ratings and Textual Content](https://dl.acm.org/doi/abs/10.1145/3534678.3539474), Nhu-Thuat Tran, Hady W. Lauw. **KDD, Aug 2022** | [`[pdf]`](https://dl.acm.org/doi/pdf/10.1145/3534678.3539474)  
- [Graph Neural Network for Tag Ranking in Tag-enhanced Video Recommendation](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3340531.3416021) - Qi Liu, Ruobing Xie, Lei Chen, Shukai Liu, Ke Tu, Peng Cui, Bo Zhang, Leyu Lin. **CIKM, Oct 2020** | [`[pdf]`](https://nlp.csai.tsinghua.edu.cn/~xrb/publications/CIKM-20_GraphTR.pdf) | [`[code]`](https://github.com/lqfarmer/GraphTR)
- [Leveraging Title-Abstract Attentive Semantics for Paper Recommendation](https://ojs.aaai.org//index.php/AAAI/article/view/5335) - Guibing Guo, Bowei Chen, Xiaoyan Zhang, Zhirong Liu, Zhenhua Dong, Xiuqiang He. **AAAI, Apr 2020** | [`[pdf]`](https://ojs.aaai.org/index.php/AAAI/article/download/5335/5191)
- [Interactive resource recommendation algorithm based on tag information](https://link.springer.com/article/10.1007/s11280-018-0532-y) - Qing Xie. Feng Xiong. Tian Han. Yongjian Liu. Lin Li. Zhifeng Bao. **WWW, Feb 2018**
- [Tag2Word: Using Tags to Generate Words for Content Based Tag Recommendation](https://dl.acm.org/doi/10.1145/2983323.2983682) - Yong Wu, Yuan Yao, Feng Xu, Hanghang Tong, Jian Lu. **CIKM, Oct 2016** | [`[pdf]`](https://cs.nju.edu.cn/yuanyao/static/cikm2016.pdf)


### Image-based Recommendation
- [CausalRec: Causal Inference for Visual Debiasing in Visually-Aware Recommendation](https://arxiv.org/abs/2107.02390) - Ruihong Qiu, Sen Wang, Zhi Chen, Hongzhi Yin, and Zi Huang. **MM, Oct 2021** | [`[pdf]`](https://arxiv.org/pdf/2107.02390.pdf) | [`[code]`](https://github.com/RuihongQiu/cornac/tree/master/cornac/models/causalrec)
- [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://ojs.aaai.org/index.php/AAAI/article/view/9973) - Ruining He, Julian McAuley. **AAAI, Feb 2016** | [`[pdf]`](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11914/11576) | [`[code]`](https://github.com/arogers1/VBPR)


- [Image and Video Understanding for Recommendation and Spam Detection Systems](https://dl.acm.org/doi/10.1145/3394486.3406485) - Aman Gupta, Sirjan Kafle, Di Wen, Dylan Wang, Sumit Srivastava, Suhit Sinha, Nikita Gupta, Bharat Jain, Ananth Sankar, Liang Zhang. **KDD, Aug 2020**
- [Exploring the Power of Visual Features for the Recommendation of Movies](https://dl.acm.org/doi/10.1145/3320435.3320470) - Mohammad Hossein Rimaz, Mehdi Elahi, Farshad Bakhshandegan Moghadam, Christoph Trattner, Reza Hosseini, Marko Tkalčič . **UMAP, Jun 2019** | [`[pdf]`](https://www.christophtrattner.com/pubs/UMAP2019.pdf)
- [Visually-Aware Personalized Recommendation using Interpretable Image Representations](https://arxiv.org/abs/1806.09820) -Charles Packer, Julian McAuley, Arnau Ramisa. **arxiv, 2018**  | [`[pdf]`](https://arxiv.org/pdf/1806.09820.pdf)
- [DeepStyle: Learning User Preferences for Visual Recommendation](https://dl.acm.org/doi/10.1145/3077136.3080658) -Qiang Liu, Shu Wu, Liang Wang. **SIGIR, Aug 2017** | [`[pdf]`](http://shuwu.name/sw/DeepStyle.pdf)


### Public Datasets (Link verified :heavy_check_mark:)
- [Amazon product data 2014] (http://jmcauley.ucsd.edu/data/amazon/links.html)
  - [Features processed version of Baby/Sports/Elec etc.](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)  
- [Amazon product data 2018] (https://nijianmo.github.io/amazon/index.html)


