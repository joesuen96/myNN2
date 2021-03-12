#include <stdio.h>                                  //����ı���Ϣ					printf
#include <stdlib.h> //�������������̬�����ڴ� 		    rand��malloc
#include <time.h>   //���������						srand(time(NULL))
#include <string.h> //memset���ٰ��ֽڳ�ʼ������		memset
#include <math.h>   //ָ������						exp(x)


#define TRAIN_IMAGES_NUM 60000                      //ѵ����ͼƬ����
#define TEST_IMAGES_NUM 10000                       //���Լ�ͼƬ����
#define LEARNING_RATE 0.3                           //ѧϰ��
#define LAYERS_NUM 4                                //���������
#define INPUT_LAYERS_SIZE 784                       //�������Ԫ����
#define HIDDEN_LAYERS_SIZE 50                       //��������Ԫ����
#define OUTPUT_LAYERS_SIZE 10                       //�������Ԫ����
#define EPOCHS 50                                   //ѵ������


//���弤���
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}


//����-1~1�����������
double randomizeWeightOrBias()
{
    return 2 * (double) rand() / RAND_MAX - 1.0; // NOLINT(cert-msc50-cpp)
}


//��Ԫ�ṹ�嶨��
typedef struct Neuron
{
    double bias;                                         //ƫ��ֵ
    double z;                                            //��Ȩ��
    double activation;                                   //����ֵ
    double *weight;                                      //ָ����ǰһ��������Ԫ������Ȩ�������ָ��
    double partialDerivativeOfLossfuncToBias;            //��ʧ�����Ե�ǰ��Ԫƫ�õ�ƫ���� ?C/?b  Partial derivative of loss function to current neuron bias
} NEURON;


//�񾭲�ṹ�嶨��
typedef struct Layer
{
    int numOfNeurons;//�ò����Ԫ��
    NEURON *neurons;//ָ��ǰ����Ԫ�����ָ��
} LAYER;


//������ṹ�嶨��
typedef struct NNet
{
    int numOfLayers;//�������еĲ���
    LAYER *layers;//ָ��������� �����ָ��
} NNET;


void loadTrainSet(FILE **fppImg, FILE **fppLabel)
{
    /*-----------------------��ѵ�����ļ�------------------------*/
    /*
        train-images.idx3-ubyte ѵ����ͼ��������ļ�
        train-labels.idx1-ubyte ѵ������ǩ�������ļ�
    */
    int magicNum, picNum, pixelRow, pixelCol;
    int lMagicNum, labelNum;

    *fppImg = fopen("./train-images.idx3-ubyte", "rb");
    if (!*fppImg)
    {
        printf("unable open train-images.idx3-ubyte���޷���ѵ����ͼ���ļ�\n");
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
        printf("unable open train-labels.idx1-ubyte���޷���ѵ������ǩ�ļ�\n");
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
    /*----------------------�����Լ��ļ�-------------------------*/

    /*
        t10k-images.idx3-ubyte ���Լ�ͼ��������ļ�
        t10k-labels.idx1-ubyte ���Լ���ǩ�������ļ�
    */

    int tMagicNum, tPicNum, tPixelRow, tPixelCol;
    int tLMagicNum, tLabelNum;

    *tFppImg = fopen("./t10k-images.idx3-ubyte", "rb");
    if (!*tFppImg)
    {
        printf("unable open t10k-images.idx3-ubyte���޷��򿪲��Լ�ͼ���ļ�\n");
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
        printf("unable open t10k-labels.idx1-ubyte���޷��򿪲��Լ���ǩ�ļ�\n");
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


//arrOfNumOfNeuronsOfEachLayersΪ�洢������Ԫ����������
//���������񾭲�����ڴ棬��������Ԫ�����ڴ棬����Ȩ����������ڴ沢�������
void initNeuronNet(NNET *nnet, int numOfLayers, const int *arrOfEachLayersNeuronsSize)
{
    nnet->numOfLayers = numOfLayers;
    //��layers���鶯̬�����ڴ�
    nnet->layers = (LAYER *) malloc(sizeof(LAYER) * numOfLayers);

    //��ÿ�����Ԫ���鶯̬�����ڴ�
    for (int i = 0; i < numOfLayers; i++)
    {
        nnet->layers[i].numOfNeurons = arrOfEachLayersNeuronsSize[i];
        nnet->layers[i].neurons = (NEURON *) malloc(sizeof(NEURON) * arrOfEachLayersNeuronsSize[i]);
    }

    //�ӵڶ��㿪ʼ��ʼ��Ȩ��
    for (int i = 1; i < nnet->numOfLayers; i++)
    {


        for (int j = 0; j < nnet->layers[i].numOfNeurons; j++)
        {
            //ÿ���ÿ����Ԫ��Ȩֵ���� ��̬�����ڴ� ��СΪ��һ�����Ԫ����
            nnet->layers[i].neurons[j].weight = (double *) malloc(sizeof(double) * nnet->layers[i - 1].numOfNeurons);

            //ÿ���ÿ����Ԫ��ƫ�������
            nnet->layers[i].neurons[j].bias = randomizeWeightOrBias();

            for (int k = 0; k < nnet->layers[i - 1].numOfNeurons; k++)
            {
                double weight = randomizeWeightOrBias();
                nnet->layers[i].neurons[j].weight[k] = weight;
            }
        }
    }
}


//����������������򴫲�
void forwardPropWithInput(NNET *nnet, const double *inputs)
{
    for (int i = 0; i < nnet->layers[0].numOfNeurons; i++)
    {
        //��������Ԫ����ֵ��ʼ��
        nnet->layers[0].neurons[i].activation = inputs[i];
    }

    //�ӵڶ��㿪ʼ��ʼ��
    for (int i = 1; i < nnet->numOfLayers; i++)
    {

        for (int j = 0; j < nnet->layers[i].numOfNeurons; j++)
        {
            //��Ȩ��
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


//��������ֵ ���򴫲� ���Ҹ���Ȩֵ����
void backProp(NNET *nnet, const double *targets)
{
    //numΪ���һ�����Ԫ��
    int numOfNeuronsOfLastLayer = nnet->layers[nnet->numOfLayers - 1].numOfNeurons;
    //iΪ���һ����Ԫ�����±�
    for (int i = 0; i < numOfNeuronsOfLastLayer; i++)
    {
        //activation�������һ����Ԫ�ļ���ֵ
        double activation = nnet->layers[nnet->numOfLayers - 1].neurons[i].activation;
        //���һ��ÿ����Ԫ�����ֵ(Cost��Ȩ��b��ƫ����) partialDerivativeOfLossfuncToBias = sigmoid(z) * ( 1 - sigmoid(z) ) * 2 * (y - activation) / OUTPUT_LAYER_SIZE
        nnet->layers[nnet->numOfLayers - 1].neurons[i].partialDerivativeOfLossfuncToBias =
                activation * (1 - activation) * (targets[i] - activation) * 2 / OUTPUT_LAYERS_SIZE;
        //����Ȩֵ
        nnet->layers[nnet->numOfLayers - 1].neurons[i].bias +=
                LEARNING_RATE * nnet->layers[nnet->numOfLayers - 1].neurons[i].partialDerivativeOfLossfuncToBias;
    }

    //iΪ��ǰ�㣬�Ӻ���ǰ
    for (int i = nnet->numOfLayers - 1; i > 0; i--)
    {
        //j�ӵ����ڶ������Ԫ����ʼ��jΪǰһ�����Ԫ��
        for (int j = 0; j < nnet->layers[i - 1].numOfNeurons; j++)
        {
            double sumOfPdOfActivationOfPreviousLayer = 0;
            //kΪ��ǰ����Ԫ��
            for (int k = 0; k < nnet->layers[i].numOfNeurons; k++)
            {
                //��ǰһ�㼤��ֵ��ƫ����
                sumOfPdOfActivationOfPreviousLayer += nnet->layers[i].neurons[k].weight[j] *
                                                      nnet->layers[i].neurons[k].partialDerivativeOfLossfuncToBias;

                //�Ե�ǰ���Ȩ��ƫ��������  ���µ�Ȩ��=ѧϰ��*�ۼ�(�Ե�ǰ��Ȩ�ص�ƫ����(=��ƫ��b��ƫ����*ǰһ��ļ���ֵ))
                nnet->layers[i].neurons[k].weight[j] +=
                        LEARNING_RATE * nnet->layers[i].neurons[k].partialDerivativeOfLossfuncToBias *
                        nnet->layers[i - 1].neurons[j].activation;

            }
            //ǰһ����Ԫ�ļ���ֵ
            double activation = nnet->layers[i - 1].neurons[j].activation;
            //ǰһ����Ԫ��ƫ�õĵ����ĸ���
            nnet->layers[i - 1].neurons[j].partialDerivativeOfLossfuncToBias =
                    activation * (1 - activation) * sumOfPdOfActivationOfPreviousLayer;
            nnet->layers[i - 1].neurons[j].bias +=
                    LEARNING_RATE * nnet->layers[i - 1].neurons[j].partialDerivativeOfLossfuncToBias;
        }

    }
}


//�����ݼ��ж�ͼƬ��������
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


//�����ݼ��ж���ǩ��������
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


//�洢ģ��Ȩ������
void saveModelData(NNET *nnet, FILE *fpModel)
{
    fpModel = fopen("./model.dat", "w+b");
    //�ӵڶ��㿪ʼ����Ȩ������
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


//��ȡģ��Ȩ������
void readModelData(NNET *nnet, FILE *fpModel)
{
//    fpModel = fopen("./model.dat", "rb");
    //�ӵڶ��㿪ʼ����Ȩ������
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


//��֤ģ�ͶԲ��Լ�����ȷ��
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
        for (int j = 0; j < TRAIN_IMAGES_NUM; j++)//��ѵ����ÿ��ͼ��
        {
//            double inputs[INPUT_LAYERS_SIZE];
//            for (int i = 0; i < INPUT_LAYERS_SIZE; i++)
//            {
//                //����ѵ�����ļ������ڴ�����ݴ浽��ʱ������
//                inputs[i] = bufferArrOfImg[j][i];
//            }

            double targets[OUTPUT_LAYERS_SIZE];
            memset(targets, 0, sizeof(double) * OUTPUT_LAYERS_SIZE);//���������ʱ���������ݰ��ֽ���0
            targets[bufferArrOfLabel[j]] = 1.0;//������ͼƬ��ǩ��Ӧ�������Ŀ����1


            //���������ǰ�򴫲�
            forwardPropWithInput(net, bufferArrOfImg[j]);
            //����Ŀ���ǩ�������򴫲�
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
                        printf("[��]%d:   %.10lf\n", i, net->layers[3].neurons[i].activation);
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
        //ÿѵ����һ��ѵ���� ����һ��ģ������
        saveModelData(net, fpModel);

        //ÿѵ����һ��ѵ���� ���һ�ΰ���ģ�����ݶ�Ӧ���Լ�����ȷ��
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
//    /*-----------------------��ѵ�����ļ�------------------------*/
//    /*
//        train-images.idx3-ubyte ѵ����ͼ��������ļ�
//        train-labels.idx1-ubyte ѵ������ǩ�������ļ�
//    */
//    int magicNum, picNum, pixelRow, pixelCol;
//    int lMagicNum, labelNum;
//
//    FILE *fpImg = fopen("./train-images.idx3-ubyte", "rb");
//    if (!fpImg)
//    {
//        printf("unable open train-images.idx3-ubyte���޷���ѵ����ͼ���ļ�\n");
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
//        printf("unable open train-labels.idx1-ubyte���޷���ѵ������ǩ�ļ�\n");
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
//    /*----------------------�����Լ��ļ�-------------------------*/
//
//    /*
//        t10k-images.idx3-ubyte ���Լ�ͼ��������ļ�
//        t10k-labels.idx1-ubyte ���Լ���ǩ�������ļ�
//    */
//    int tMagicNum, tPicNum, tPixelRow, tPixelCol;
//    int tLMagicNum, tLabelNum;
//
//    FILE *tFpImg = fopen("./t10k-images.idx3-ubyte", "rb");
//    if (!tFpImg)
//    {
//        printf("unable open t10k-images.idx3-ubyte���޷��򿪲��Լ�ͼ���ļ�\n");
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
//        printf("unable open t10k-labels.idx1-ubyte���޷��򿪲��Լ���ǩ�ļ�\n");
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

    //�����洢ѵ������ͼƬ�Ҷ�ֵ������
    double **bufferArrOfTrainImages = (double **) malloc(sizeof(double *) * TRAIN_IMAGES_NUM); //bufferArrOfTrainImages[60000][28*28]
    for (int i = 0; i < TRAIN_IMAGES_NUM; i++)//60000��ѵ����ͼƬ
    {
        //ÿ��ͼƬ�����ٳ�28*28��С������
        bufferArrOfTrainImages[i] = (double *) malloc(sizeof(double) * INPUT_LAYERS_SIZE);
    }
    initBufferArrOfImage(fpImg, bufferArrOfTrainImages, TRAIN_IMAGES_NUM);    //���ļ�ָ��fpimg�ж�ȡ�Ҷ�ֵ����ά����bufferArrOfImg��
    printf("ѵ����ͼ���ļ���ȡ���\n");


    //�����洢ѵ�����б�ǩ������
    int *bufferArrOfTrainLabels = (int *) malloc(sizeof(int) * TRAIN_IMAGES_NUM);//bufferArrOfTrainLabels[60000]
    initBufferArrOfLabel(fpLabel, bufferArrOfTrainLabels, TRAIN_IMAGES_NUM); //���ļ�ָ��fpLabel�ж�ȡ��ǩֵ����ά����bufferArrOfLabel��
    printf("ѵ������ǩ�ļ���ȡ���\n");


    //�����洢���Լ���ͼƬ�Ҷ�ֵ������
    double **bufferArrOfTestImages = (double **) malloc(sizeof(double *) * TEST_IMAGES_NUM);
    for (int i = 0; i < TEST_IMAGES_NUM; i++)
    {
        bufferArrOfTestImages[i] = (double *) malloc(sizeof(double) * INPUT_LAYERS_SIZE);
    }
    initBufferArrOfImage(tFpImg, bufferArrOfTestImages, TEST_IMAGES_NUM);
    printf("���Լ�ͼ���ļ���ȡ���\n");


    //�����洢���Լ��б�ǩ������
    int *bufferArrOfTestLabels = (int *) malloc(sizeof(int) * TEST_IMAGES_NUM);
    initBufferArrOfLabel(tFpLabel, bufferArrOfTestLabels, TEST_IMAGES_NUM);
    printf("���Լ���ǩ�ļ���ȡ���\n");




    //����������ڴ�ռ�
    NNET *net = (NNET *) malloc(sizeof(NNET));


    //��Ÿ��� ��Ԫ���� �� ����
    int arrOfEachLayersNeuronsSize[LAYERS_NUM] = {INPUT_LAYERS_SIZE,
                                                  HIDDEN_LAYERS_SIZE,
                                                  HIDDEN_LAYERS_SIZE,
                                                  OUTPUT_LAYERS_SIZE};


    //��ʼ�����������
    srand((unsigned int) time(NULL)); // NOLINT(cert-msc51-cpp)
    //��ʼ���������ڴ�ռ䲢�ҳ�ʼ�������㵽�������Ԫ��ƫ��
    initNeuronNet(net, LAYERS_NUM, arrOfEachLayersNeuronsSize);
    printf("�������ʼ�����\n");


    //��ʼ����������ģ�����ݵ��ļ�ָ�루�ϵ���ѵ����ؽ�����ƣ�
    FILE *fpModel = NULL;


    //�������ģ���ļ��Ƿ���ڵ��ļ�ָ��fpModelData
    FILE *fpModelData = fopen("./model.dat", "rb");
    if (fpModelData)
    {
        //�������ģ������ ��ȡģ���и���Ԫ����һ���Ȩ�غ�ƫ�� ֮��ر��ļ�
        readModelData(net, fpModelData);
        printf("��⵽ģ�����������ϵ���ѵ\n\n");
        continueTrainAtBreak(net,
                             bufferArrOfTrainImages,
                             bufferArrOfTestImages,
                             bufferArrOfTrainLabels,
                             bufferArrOfTestLabels,
                             fpModel);
    }
    else
    {
        //�ϵ���ѵ����
        continueTrainAtBreak(net,
                             bufferArrOfTrainImages,
                             bufferArrOfTestImages,
                             bufferArrOfTrainLabels,
                             bufferArrOfTestLabels,
                             fpModel);
    }


    //�ͷ��ڴ�
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