#include <stdio.h>                                  //输出文本信息					printf
#include <stdlib.h> //生成随机数、动态开辟内存 		    rand、malloc
#include <time.h>   //随机数种子						srand(time(NULL))
#include <string.h> //memset快速按字节初始化数组		memset
#include <math.h>   //指数函数						exp(x)


#define TRAIN_IMAGES_NUM 60000                      //训练集图片数量
#define TEST_IMAGES_NUM 10000                       //测试集图片数量
#define LEARNING_RATE 0.3                           //学习率
#define LAYERS_NUM 4                                //神经网络层数
#define INPUT_LAYERS_SIZE 784                       //输入层神经元数量
#define HIDDEN_LAYERS_SIZE 50                       //隐含层神经元数量
#define OUTPUT_LAYERS_SIZE 10                       //输出层神经元数量
#define EPOCHS 50                                   //训练轮数


//定义激活函数
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}


//生成-1~1的随机浮点数
double randomizeWeightOrBias()
{
    return 2 * (double) rand() / RAND_MAX - 1.0; // NOLINT(cert-msc50-cpp)
}


//神经元结构体定义
typedef struct Neuron
{
    double bias;                                         //偏置值
    double z;                                            //加权和
    double activation;                                   //激活值
    double *weight;                                      //指向与前一层所有神经元相连的权重数组的指针
    double partialDerivativeOfLossfuncToBias;            //损失函数对当前神经元偏置的偏导数 ?C/?b  Partial derivative of loss function to current neuron bias
} NEURON;


//神经层结构体定义
typedef struct Layer
{
    int numOfNeurons;//该层的神经元数
    NEURON *neurons;//指向当前层神经元数组的指针
} LAYER;


//神经网络结构体定义
typedef struct NNet
{
    int numOfLayers;//神经网络中的层数
    LAYER *layers;//指向神经网络层 数组的指针
} NNET;


void loadTrainSet(FILE **fppImg, FILE **fppLabel)
{
    /*-----------------------读训练集文件------------------------*/
    /*
        train-images.idx3-ubyte 训练集图像二进制文件
        train-labels.idx1-ubyte 训练集标签二进制文件
    */
    int magicNum, picNum, pixelRow, pixelCol;
    int lMagicNum, labelNum;

    *fppImg = fopen("./train-images.idx3-ubyte", "rb");
    if (!*fppImg)
    {
        printf("unable open train-images.idx3-ubyte，无法打开训练集图像文件\n");
        exit(0);
    }
    else
    {
        fread(&magicNum, sizeof(int), 1, *fppImg);
        fread(&picNum, sizeof(int), 1, *fppImg);
        fread(&pixelRow, sizeof(int), 1, *fppImg);
        fread(&pixelCol, sizeof(int), 1, *fppImg);
//        fseek(fppImg, 4 * sizeof(int), SEEK_CUR);
    }

    *fppLabel = fopen("./train-labels.idx1-ubyte", "rb");
    if (!*fppLabel)
    {
        printf("unable open train-labels.idx1-ubyte，无法打开训练集标签文件\n");
        exit(0);
    }
    else
    {
        fread(&lMagicNum, sizeof(int), 1, *fppLabel);
        fread(&labelNum, sizeof(int), 1, *fppLabel);
//        fseek(fpLable, 2 * sizeof(int), SEEK_CUR);
    }
    /*----------------------------------------------------------*/

}


void loadTestSet(FILE **tFppImg, FILE **tFppLabel)
{
    /*----------------------读测试集文件-------------------------*/

    /*
        t10k-images.idx3-ubyte 测试集图像二进制文件
        t10k-labels.idx1-ubyte 测试集标签二进制文件
    */

    int tMagicNum, tPicNum, tPixelRow, tPixelCol;
    int tLMagicNum, tLabelNum;

    *tFppImg = fopen("./t10k-images.idx3-ubyte", "rb");
    if (!*tFppImg)
    {
        printf("unable open t10k-images.idx3-ubyte，无法打开测试集图像文件\n");
        exit(0);
    }
    else
    {
        fread(&tMagicNum, sizeof(int), 1, *tFppImg);
        fread(&tPicNum, sizeof(int), 1, *tFppImg);
        fread(&tPixelRow, sizeof(int), 1, *tFppImg);
        fread(&tPixelCol, sizeof(int), 1, *tFppImg);
//        fseek(tFppImg,4*sizeof(int),SEEK_CUR);
    }

    *tFppLabel = fopen("./t10k-labels.idx1-ubyte", "rb");
    if (!*tFppLabel)
    {
        printf("unable open t10k-labels.idx1-ubyte，无法打开测试集标签文件\n");
        exit(0);
    }
    else
    {
        fread(&tLMagicNum, sizeof(int), 1, *tFppLabel);
        fread(&tLabelNum, sizeof(int), 1, *tFppLabel);
//        fseek(tFpLable, 2 * sizeof(int), SEEK_CUR);
    }

    /*------------------------------------------------------------*/

}


//arrOfNumOfNeuronsOfEachLayers为存储各层神经元个数的数组
//包括给各神经层分配内存，给各层神经元分配内存，给各权重网络分配内存并且随机化
void initNeuronNet(NNET *nnet, int numOfLayers, const int *arrOfEachLayersNeuronsSize)
{
    nnet->numOfLayers = numOfLayers;
    //给layers数组动态分配内存
    nnet->layers = (LAYER *) malloc(sizeof(LAYER) * numOfLayers);

    //给每层的神经元数组动态分配内存
    for (int i = 0; i < numOfLayers; i++)
    {
        nnet->layers[i].numOfNeurons = arrOfEachLayersNeuronsSize[i];
        nnet->layers[i].neurons = (NEURON *) malloc(sizeof(NEURON) * arrOfEachLayersNeuronsSize[i]);
    }

    //从第二层开始初始化权重
    for (int i = 1; i < nnet->numOfLayers; i++)
    {


        for (int j = 0; j < nnet->layers[i].numOfNeurons; j++)
        {
            //每层的每个神经元的权值数组 动态分配内存 大小为上一层的神经元个数
            nnet->layers[i].neurons[j].weight = (double *) malloc(sizeof(double) * nnet->layers[i - 1].numOfNeurons);

            //每层的每个神经元的偏置随机化
            nnet->layers[i].neurons[j].bias = randomizeWeightOrBias();

            for (int k = 0; k < nnet->layers[i - 1].numOfNeurons; k++)
            {
                double weight = randomizeWeightOrBias();
                nnet->layers[i].neurons[j].weight[k] = weight;
            }
        }
    }
}


//输入层输入数据正向传播
void forwardPropWithInput(NNET *nnet, const double *inputs)
{
    for (int i = 0; i < nnet->layers[0].numOfNeurons; i++)
    {
        //输入层各神经元激活值初始化
        nnet->layers[0].neurons[i].activation = inputs[i];
    }

    //从第二层开始初始化
    for (int i = 1; i < nnet->numOfLayers; i++)
    {

        for (int j = 0; j < nnet->layers[i].numOfNeurons; j++)
        {
            //加权和
            double z = 0;
            for (int k = 0; k < nnet->layers[i - 1].numOfNeurons; k++)
            {
                double weight = nnet->layers[i].neurons[j].weight[k];
                z += nnet->layers[i - 1].neurons[k].activation * weight;
            }
            nnet->layers[i].neurons[j].z = z + nnet->layers[i].neurons[j].bias;
            nnet->layers[i].neurons[j].activation = sigmoid(nnet->layers[i].neurons[j].z);
        }

    }
}


//根据期望值 反向传播 并且更新权值网络
void backProp(NNET *nnet, const double *targets)
{
    //num为最后一层的神经元数
    int numOfNeuronsOfLastLayer = nnet->layers[nnet->numOfLayers - 1].numOfNeurons;
    //i为最后一层神经元数组下标
    for (int i = 0; i < numOfNeuronsOfLastLayer; i++)
    {
        //activation等于最后一层神经元的激活值
        double activation = nnet->layers[nnet->numOfLayers - 1].neurons[i].activation;
        //最后一层每个神经元的误差值(Cost对权重b的偏导数) partialDerivativeOfLossfuncToBias = sigmoid(z) * ( 1 - sigmoid(z) ) * 2 * (y - activation) / OUTPUT_LAYER_SIZE
        nnet->layers[nnet->numOfLayers - 1].neurons[i].partialDerivativeOfLossfuncToBias =
                activation * (1 - activation) * (targets[i] - activation) * 2 / OUTPUT_LAYERS_SIZE;
        //更新权值
        nnet->layers[nnet->numOfLayers - 1].neurons[i].bias +=
                LEARNING_RATE * nnet->layers[nnet->numOfLayers - 1].neurons[i].partialDerivativeOfLossfuncToBias;
    }

    //i为当前层，从后往前
    for (int i = nnet->numOfLayers - 1; i > 0; i--)
    {
        //j从倒数第二层的神经元数开始，j为前一层的神经元数
        for (int j = 0; j < nnet->layers[i - 1].numOfNeurons; j++)
        {
            double sumOfPdOfActivationOfPreviousLayer = 0;
            //k为当前层神经元数
            for (int k = 0; k < nnet->layers[i].numOfNeurons; k++)
            {
                //对前一层激活值的偏导数
                sumOfPdOfActivationOfPreviousLayer += nnet->layers[i].neurons[k].weight[j] *
                                                      nnet->layers[i].neurons[k].partialDerivativeOfLossfuncToBias;

                //对当前层的权重偏导数更新  更新的权重=学习率*累加(对当前层权重的偏导数(=对偏置b的偏导数*前一层的激活值))
                nnet->layers[i].neurons[k].weight[j] +=
                        LEARNING_RATE * nnet->layers[i].neurons[k].partialDerivativeOfLossfuncToBias *
                        nnet->layers[i - 1].neurons[j].activation;

            }
            //前一层神经元的激活值
            double activation = nnet->layers[i - 1].neurons[j].activation;
            //前一层神经元对偏置的导数的更新
            nnet->layers[i - 1].neurons[j].partialDerivativeOfLossfuncToBias =
                    activation * (1 - activation) * sumOfPdOfActivationOfPreviousLayer;
            nnet->layers[i - 1].neurons[j].bias +=
                    LEARNING_RATE * nnet->layers[i - 1].neurons[j].partialDerivativeOfLossfuncToBias;
        }

    }
}


//从数据集中读图片到数组中
void initBufferArrOfImage(FILE *fpImg, double **bufferArr, int numOfImgs)
{
    unsigned char *tmpBufferOfImg = (unsigned char *) malloc(sizeof(unsigned char) * INPUT_LAYERS_SIZE);

    for (int i = 0; i < numOfImgs; i++)
    {
        fread(tmpBufferOfImg, sizeof(unsigned char), INPUT_LAYERS_SIZE, fpImg);

        for (int j = 0; j < INPUT_LAYERS_SIZE; j++)
        {
            bufferArr[i][j] = tmpBufferOfImg[j] / 255.0;
        }
    }
    free(tmpBufferOfImg);

}


//从数据集中读标签到数组中
void initBufferArrOfLabel(FILE *fpLabel, int *bufferArr, int numOfLabels)
{
    unsigned char *tmpBufferOfLabel = (unsigned char *) malloc(sizeof(unsigned char) * numOfLabels);
    //memset(tmpBufferOfLabel, 0, sizeof(unsigned char) * TRAIN_IMAGES_NUM);

    fread(tmpBufferOfLabel, sizeof(unsigned char), numOfLabels, fpLabel);
    for (int i = 0; i < numOfLabels; i++)
    {
        bufferArr[i] = tmpBufferOfLabel[i];
    }
    free(tmpBufferOfLabel);

}


//存储模型权重数据
void saveModelData(NNET *nnet, FILE *fpModel)
{
    fpModel = fopen("./model.dat", "w+b");
    //从第二层开始才有权重网络
    for (int i = 1; i < nnet->numOfLayers; i++)
    {
        for (int j = 0; j < nnet->layers[i].numOfNeurons; j++)
        {
            for (int k = 0; k < nnet->layers[i - 1].numOfNeurons; k++)
            {
                fwrite(&(nnet->layers[i].neurons[j].weight[k]), sizeof(double), 1, fpModel);
            }
            fwrite(&(nnet->layers[i].neurons[j].bias), sizeof(double), 1, fpModel);
        }
    }
    fclose(fpModel);
}


//读取模型权重数据
void readModelData(NNET *nnet, FILE *fpModel)
{
//    fpModel = fopen("./model.dat", "rb");
    //从第二层开始才有权重网络
    for (int i = 1; i < nnet->numOfLayers; i++)
    {
        for (int j = 0; j < nnet->layers[i].numOfNeurons; j++)
        {
            for (int k = 0; k < nnet->layers[i - 1].numOfNeurons; k++)
            {
                fread(&(nnet->layers[i].neurons[j].weight[k]), sizeof(double), 1, fpModel);
            }
            fread(&(nnet->layers[i].neurons[j].bias), sizeof(double), 1, fpModel);
        }
    }
    fclose(fpModel);
}


//验证模型对测试集的正确率
double accuracyRate(NNET *nnet, double **tBufferArrOfImg, const int *tBufferArrOfLabel)
{
    int cntRight = 0;
    for (int i = 0; i < TEST_IMAGES_NUM; i++)
    {
        double tInputs[INPUT_LAYERS_SIZE];
        for (int j = 0; j < INPUT_LAYERS_SIZE; j++)
        {
            tInputs[j] = tBufferArrOfImg[i][j];
        }

        forwardPropWithInput(nnet, tInputs);

        double max = nnet->layers[nnet->numOfLayers - 1].neurons[0].activation;
        int guessNum = 0;
        for (int j = 0; j < OUTPUT_LAYERS_SIZE; j++)
        {
            if (nnet->layers[nnet->numOfLayers - 1].neurons[j].activation > max)
            {
                max = nnet->layers[nnet->numOfLayers - 1].neurons[j].activation;
                guessNum = j;
            }
        }

        if (guessNum == tBufferArrOfLabel[i])
        {
            cntRight++;
        }
    }
    return (double) cntRight / TEST_IMAGES_NUM;
}


void continueTrainAtBreak(NNET *net,
                          double **bufferArrOfImg,
                          double **tBufferArrOfImg,
                          int *bufferArrOfLabel,
                          int *tBufferArrOfLabel,
                          FILE *fpModel)
{
    for (int k = 0; k < EPOCHS; k++)
    {
        for (int j = 0; j < TRAIN_IMAGES_NUM; j++)//对训练集每张图像
        {
//            double inputs[INPUT_LAYERS_SIZE];
//            for (int i = 0; i < INPUT_LAYERS_SIZE; i++)
//            {
//                //将从训练集文件读到内存的数据存到临时变量中
//                inputs[i] = bufferArrOfImg[j][i];
//            }

            double targets[OUTPUT_LAYERS_SIZE];
            memset(targets, 0, sizeof(double) * OUTPUT_LAYERS_SIZE);//将输出层临时变量的数据按字节置0
            targets[bufferArrOfLabel[j]] = 1.0;//将该张图片标签对应的输出层目标置1


            //输入层数据前向传播
            forwardPropWithInput(net, bufferArrOfImg[j]);
            //根据目标标签向量反向传播
            backProp(net, targets);


            if ((j + 1) % 10000 == 0)
            {
                printf("epoch:%d , index:%d , label:%d\n",
                       k + 1,
                       j + 1,
                       bufferArrOfLabel[j]);

                for (int i = 0; i < net->layers[3].numOfNeurons; i++)
                {
                    if (i == bufferArrOfLabel[j])
                        printf("[√]%d:   %.10lf\n", i, net->layers[3].neurons[i].activation);
                    else
                        printf("   %d:   %.10lf\n", i, net->layers[3].neurons[i].activation);
                }
                printf("\n");


                /*printf("-------------------testImg:7------------------------\n\n");
                forwardPropWithInput(net, tbuffer);
                for (int i = 0; i < net->layers[3].numOfNeurons; i++)
                {
                    printf("%d:   %.20lf\n", i, net->layers[3].neurons[i].activation);
                }
                printf("\n");*/

            }
        }
        //每训练完一次训练集 保存一次模型数据
        saveModelData(net, fpModel);

        //每训练完一次训练集 输出一次按照模型数据对应测试集的正确率
        printf("accuracy rate:%.3lf%%\n", 100 * accuracyRate(net, tBufferArrOfImg, tBufferArrOfLabel));
    }
}


int main()
{

    FILE *fpImg = NULL;
    FILE *fpLabel = NULL;
    FILE *tFpImg = NULL;
    FILE *tFpLabel = NULL;

    loadTrainSet(&fpImg, &fpLabel);
    loadTestSet(&tFpImg, &tFpLabel);

#pragma region
//    /*-----------------------读训练集文件------------------------*/
//    /*
//        train-images.idx3-ubyte 训练集图像二进制文件
//        train-labels.idx1-ubyte 训练集标签二进制文件
//    */
//    int magicNum, picNum, pixelRow, pixelCol;
//    int lMagicNum, labelNum;
//
//    FILE *fpImg = fopen("./train-images.idx3-ubyte", "rb");
//    if (!fpImg)
//    {
//        printf("unable open train-images.idx3-ubyte，无法打开训练集图像文件\n");
//        return 0;
//    }
//    else
//    {
//        fread(&magicNum, sizeof(int), 1, fpImg);
//        fread(&picNum, sizeof(int), 1, fpImg);
//        fread(&pixelRow, sizeof(int), 1, fpImg);
//        fread(&pixelCol, sizeof(int), 1, fpImg);
////        fseek(fpImg, 4 * sizeof(int), SEEK_CUR);
//    }
//
//    FILE *fpLabel = fopen("./train-labels.idx1-ubyte", "rb");
//    if (!fpLabel)
//    {
//        printf("unable open train-labels.idx1-ubyte，无法打开训练集标签文件\n");
//        return 0;
//    }
//    else
//    {
//        fread(&lMagicNum, sizeof(int), 1, fpLabel);
//        fread(&labelNum, sizeof(int), 1, fpLabel);
////        fseek(fpLabel, 2 * sizeof(int), SEEK_CUR);
//    }
//    /*----------------------------------------------------------*/
//
//
//    /*----------------------读测试集文件-------------------------*/
//
//    /*
//        t10k-images.idx3-ubyte 测试集图像二进制文件
//        t10k-labels.idx1-ubyte 测试集标签二进制文件
//    */
//    int tMagicNum, tPicNum, tPixelRow, tPixelCol;
//    int tLMagicNum, tLabelNum;
//
//    FILE *tFpImg = fopen("./t10k-images.idx3-ubyte", "rb");
//    if (!tFpImg)
//    {
//        printf("unable open t10k-images.idx3-ubyte，无法打开测试集图像文件\n");
//        return 0;
//    }
//    else
//    {
//        fread(&tMagicNum, sizeof(int), 1, tFpImg);
//        fread(&tPicNum, sizeof(int), 1, tFpImg);
//        fread(&tPixelRow, sizeof(int), 1, tFpImg);
//        fread(&tPixelCol, sizeof(int), 1, tFpImg);
////        fseek(tFpImg,4*sizeof(int),SEEK_CUR);
//    }
//
//    FILE *tFpLabel = fopen("./t10k-labels.idx1-ubyte", "rb");
//    if (!tFpLabel)
//    {
//        printf("unable open t10k-labels.idx1-ubyte，无法打开测试集标签文件\n");
//        return 0;
//    }
//    else
//    {
//        fread(&tLMagicNum, sizeof(int), 1, tFpLabel);
//        fread(&tLabelNum, sizeof(int), 1, tFpLabel);
////        fseek(tFpLabel, 2 * sizeof(int), SEEK_CUR);
//    }
//
//    /*------------------------------------------------------------*/
#pragma endregion

    //用来存储训练集中图片灰度值的数组
    double **bufferArrOfTrainImages = (double **) malloc(sizeof(double *) * TRAIN_IMAGES_NUM); //bufferArrOfTrainImages[60000][28*28]
    for (int i = 0; i < TRAIN_IMAGES_NUM; i++)//60000张训练集图片
    {
        //每张图片都开辟出28*28大小的数组
        bufferArrOfTrainImages[i] = (double *) malloc(sizeof(double) * INPUT_LAYERS_SIZE);
    }
    initBufferArrOfImage(fpImg, bufferArrOfTrainImages, TRAIN_IMAGES_NUM);    //从文件指针fpimg中读取灰度值到二维数组bufferArrOfImg中
    printf("训练集图像文件读取完毕\n");


    //用来存储训练集中标签的数组
    int *bufferArrOfTrainLabels = (int *) malloc(sizeof(int) * TRAIN_IMAGES_NUM);//bufferArrOfTrainLabels[60000]
    initBufferArrOfLabel(fpLabel, bufferArrOfTrainLabels, TRAIN_IMAGES_NUM); //从文件指针fpLabel中读取标签值到二维数组bufferArrOfLabel中
    printf("训练集标签文件读取完毕\n");


    //用来存储测试集中图片灰度值的数组
    double **bufferArrOfTestImages = (double **) malloc(sizeof(double *) * TEST_IMAGES_NUM);
    for (int i = 0; i < TEST_IMAGES_NUM; i++)
    {
        bufferArrOfTestImages[i] = (double *) malloc(sizeof(double) * INPUT_LAYERS_SIZE);
    }
    initBufferArrOfImage(tFpImg, bufferArrOfTestImages, TEST_IMAGES_NUM);
    printf("测试集图像文件读取完毕\n");


    //用来存储测试集中标签的数组
    int *bufferArrOfTestLabels = (int *) malloc(sizeof(int) * TEST_IMAGES_NUM);
    initBufferArrOfLabel(tFpLabel, bufferArrOfTestLabels, TEST_IMAGES_NUM);
    printf("测试集标签文件读取完毕\n");




    //给网络分配内存空间
    NNET *net = (NNET *) malloc(sizeof(NNET));


    //存放各层 神经元个数 的 数组
    int arrOfEachLayersNeuronsSize[LAYERS_NUM] = {INPUT_LAYERS_SIZE,
                                                  HIDDEN_LAYERS_SIZE,
                                                  HIDDEN_LAYERS_SIZE,
                                                  OUTPUT_LAYERS_SIZE};


    //初始化随机数种子
    srand((unsigned int) time(NULL)); // NOLINT(cert-msc51-cpp)
    //初始化神经网络内存空间并且初始化隐含层到输出层神经元的偏置
    initNeuronNet(net, LAYERS_NUM, arrOfEachLayersNeuronsSize);
    printf("神经网络初始化完毕\n");


    //初始化用来保存模型数据的文件指针（断点续训功能亟待完善）
    FILE *fpModel = NULL;


    //用来检测模型文件是否存在的文件指针fpModelData
    FILE *fpModelData = fopen("./model.dat", "rb");
    if (fpModelData)
    {
        //如果存在模型数据 读取模型中各神经元对上一层的权重和偏置 之后关闭文件
        readModelData(net, fpModelData);
        printf("检测到模型数据启动断点续训\n\n");
        continueTrainAtBreak(net,
                             bufferArrOfTrainImages,
                             bufferArrOfTestImages,
                             bufferArrOfTrainLabels,
                             bufferArrOfTestLabels,
                             fpModel);
    }
    else
    {
        //断点续训功能
        continueTrainAtBreak(net,
                             bufferArrOfTrainImages,
                             bufferArrOfTestImages,
                             bufferArrOfTrainLabels,
                             bufferArrOfTestLabels,
                             fpModel);
    }


    //释放内存
    free(bufferArrOfTrainImages);
    free(bufferArrOfTrainLabels);
    free(bufferArrOfTestImages);
    free(bufferArrOfTestLabels);

    fclose(fpImg);
    fclose(fpLabel);
    fclose(tFpImg);
    fclose(tFpLabel);

    getchar();
    return 0;
}