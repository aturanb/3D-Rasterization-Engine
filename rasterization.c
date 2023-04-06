/*
LINK TO THE VIDEO: https://www.youtube.com/shorts/Xty4EAEbKcQ
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
//#define min(a,b) (((a) < (b)) ? (a) : (b))

#define NORMALS

#define WIDTH 1000
#define HEIGHT 1000

typedef struct 
{
    double lightDir[3]; // The direction of the light source
    double Ka;           // The coefficient for ambient lighting.
    double Kd;           // The coefficient for diffuse lighting.
    double Ks;           // The coefficient for specular lighting.
    double alpha;        // The exponent term for specular lighting.
} LightingParameters;

typedef struct
{
    double          near, far;
    double          angle;
    double          position[3];
    double          focus[3];
    double          up[3];
} Camera;

typedef struct
{
    double          U[3];
    double          V[3];
    double          W[3];
    double          O[3];
} CameraFrame;

//Image structure that contains the size and the data of the image
typedef struct image{
    int width;
    int height;
    unsigned char *data;
    double depthBuffer[HEIGHT][WIDTH];
} Image;

//Triangle structure that contains the triangle vertices and color
typedef struct
{
   double         X[3];
   double         Y[3];
   double         Z[3];
   double         color[3][3]; // color[2][0] is for V2, red channel
#ifdef NORMALS
   double         normals[3][3]; // normals[2][0] is for V2, x-component
#endif
    double shading[3];
} Triangle;

//Triangle list that contains the triangles
typedef struct
{
   int numTriangles;
   Triangle *triangles;
} TriangleList;

typedef struct
{
    double          A[4][4];     // A[i][j] means row i, column j
} Matrix;

/* FUNCTION DECLARATIONS */
void rasterize(Triangle *triangle, Image *canvas, int triangleid);
void assignPixels(Image *canvas, int colMin, int colMax, int rowMin, int rowMax, unsigned char R, unsigned char G, unsigned char B);
void TransformAndRenderTriangles(Camera c, TriangleList* t1, Image* canvas, LightingParameters lp);
void saveImage(Image *canvas, int frame);
void swapDouble(double *a, double *b);
void swapInt(int *a, int *b);
void printTriangle(Triangle *triangle);
void set_initial_depth(Image *canvas);
double interpolateScalar(double X, double A, double fA, double B, double fB);
Matrix GetViewTransform(Camera c);
Matrix GetCameraTransform(Camera c);
Matrix GetDeviceTransform(Camera c);
//Error checking functions:
int errorFlatTriangleCheck(Triangle* triangle);
//Provided Starter Functions:
char *Read3Numbers(char *tmp, double *v1, double *v2, double *v3);
Camera GetCamera(int frame, int nframes);
Matrix ComposeMatrices(Matrix M1, Matrix M2);
void TransformPoint(Matrix m, const double *ptIn, double *ptOut);
double SineParameterize(int curFrame, int nFrames, int ramp);
TriangleList *Get3DTriangles();
void PrintMatrix(Matrix m);
double C441(double f);
double F441(double f);

void divideByMagnitude(double* A, double magnitude, int size){
    for(int i = 0; i < size; i++){
        A[i] = A[i]/magnitude;
    }
}


double dot_product(double* A, double* B, int size) {
  double result = 0;
  for (int i = 0; i < size; i++) {
    result += A[i] * B[i];
  }
  return result;
}

LightingParameters GetLighting(Camera c)
{
    LightingParameters lp;
    lp.Ka = 0.3;
    lp.Kd = 0.7;
    lp.Ks = 2.8;
    lp.alpha = 50.5;
    lp.lightDir[0] = c.position[0]-c.focus[0];
    lp.lightDir[1] = c.position[1]-c.focus[1];
    lp.lightDir[2] = c.position[2]-c.focus[2];
    double mag = sqrt(lp.lightDir[0]*lp.lightDir[0]
                    + lp.lightDir[1]*lp.lightDir[1]
                    + lp.lightDir[2]*lp.lightDir[2]);
    if (mag > 0)
    {
        lp.lightDir[0] /= mag;
        lp.lightDir[1] /= mag;
        lp.lightDir[2] /= mag;
    }

    return lp;
}

int main() {

    // Initialize a new image structure
    Image canvas;
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    
    //Fill the triangle list with triangles
    TriangleList* t1 = Get3DTriangles();
    

    for(int i = 0; i < 1000; i++){
        
        /* UNCOMMENT: To generate 4 frames rather than 1000 */
        //if (i % 250 != 0)
        //   continue;

        /* INITIALIZE */
        canvas.data = (unsigned char*)malloc(canvas.width * canvas.height * 3);
        assignPixels(&canvas, 0, canvas.width, 0, canvas.height, 0, 0, 0);
        set_initial_depth(&canvas);
        
        /* BUILD THE IMAGE */
        Camera c = GetCamera(i, 1000);
        LightingParameters lp = GetLighting(c);
        //printf("Calling GetLighting and got a light direction of %f, %f, %f\n", lp.lightDir[0], lp.lightDir[1], lp.lightDir[2]);
        TransformAndRenderTriangles(c, t1, &canvas, lp);
        
        /* SAVE  */
        saveImage(&canvas, i);
    }

    return 0;
}

void TransformAndRenderTriangles(Camera c, TriangleList* t1, Image* canvas, LightingParameters lp){
    
    /* GET TRANSFORM MATRICES AND COMPOSE THEM IN A SINGLE MATRIX */
    Matrix cameraTransfrom = GetCameraTransform(c);
    Matrix viewTransform = GetViewTransform(c);
    Matrix deviceTransform = GetDeviceTransform(c);
    Matrix tmp = ComposeMatrices(cameraTransfrom, viewTransform);
    Matrix result = ComposeMatrices(tmp, deviceTransform);

    double pointOut[3];
    double pointIn[3];
    for(int triangle = 0; triangle < t1->numTriangles; triangle++){
        Triangle tempT;
        //printf("Working on triangle %d\n", triangle);
        //printf("	(%f, %f, %f), color = (%f,%f,%f)\n", t1->triangles[triangle].X[0], t1->triangles[triangle].Y[0], t1->triangles[triangle].Z[0], t1->triangles[triangle].color[0][0], t1->triangles[triangle].color[0][1], t1->triangles[triangle].color[0][2]);
        //printf("	(%f, %f, %f), color = (%f,%f,%f)\n", t1->triangles[triangle].X[1], t1->triangles[triangle].Y[1], t1->triangles[triangle].Z[1], t1->triangles[triangle].color[1][0], t1->triangles[triangle].color[1][1], t1->triangles[triangle].color[1][2]);
        //printf("	(%f, %f, %f), color = (%f,%f,%f)\n", t1->triangles[triangle].X[2], t1->triangles[triangle].Y[2], t1->triangles[triangle].Z[2], t1->triangles[triangle].color[2][0], t1->triangles[triangle].color[2][1], t1->triangles[triangle].color[2][2]);
        for(int i = 0; i < 3; i++){
            double viewDirection[3];
            double diffuse;

            //printf(" Working on vertex %d\n", i);
            
            viewDirection[0] = c.position[0] - t1->triangles[triangle].X[i];
            viewDirection[1] = c.position[1] - t1->triangles[triangle].Y[i];
            viewDirection[2] = c.position[2] - t1->triangles[triangle].Z[i];
            
            double magnitude = sqrt(viewDirection[0]*viewDirection[0] + viewDirection[1]*viewDirection[1] + viewDirection[2]*viewDirection[2]);
            divideByMagnitude(viewDirection, magnitude, 3);
            
            //printf("		View dir for pt %f, %f, %f is %f, %f, %f\n", t1->triangles[triangle].X[i], t1->triangles[triangle].Y[i], t1->triangles[triangle].Z[i], viewDirection[0], viewDirection[1], viewDirection[2]);
            //printf("		Normal is %f, %f, %f\n", t1->triangles[triangle].normals[i][0], t1->triangles[triangle].normals[i][1], t1->triangles[triangle].normals[i][2]);
            
            //Diffuse
            double ldotn = dot_product(lp.lightDir, t1->triangles[triangle].normals[i], 3);
            //printf("		LdotN is %f\n", ldotn);
            diffuse = lp.Kd * fmax(0, ldotn);
            //printf("	Diffuse is %f\n", diffuse);
            
            //Specular:
            double reflectionVector[3]; 
            reflectionVector[0] = 2 * ldotn * t1->triangles[triangle].normals[i][0] - lp.lightDir[0];
            reflectionVector[1] = 2 * ldotn * t1->triangles[triangle].normals[i][1] - lp.lightDir[1];
            reflectionVector[2] = 2 * ldotn * t1->triangles[triangle].normals[i][2] - lp.lightDir[2];
            magnitude = sqrt(reflectionVector[0]*reflectionVector[0] + reflectionVector[1]*reflectionVector[1] + reflectionVector[2]*reflectionVector[2]);
            divideByMagnitude(reflectionVector, magnitude, 3);
            
            double rDotV = dot_product(reflectionVector, viewDirection, 3);
            //printf("		Reflection vector R is %f, %f, %f\n", reflectionVector[0], reflectionVector[1], reflectionVector[2]);
            //printf("		RdotV is %f\n", rDotV);

            
            double specular = (lp.Ks*pow((fmax(0, rDotV)), lp.alpha));
            if(rDotV < 0){
                rDotV = 0;
            }
            //printf("	Specular component is %f\n", specular);

            //Total Shading:
            double total = lp.Ka + diffuse + specular;
            //printf("	Total value for vertex is %f\n", total);
            tempT.shading[i] = total;
            //Set the pointIn equal to vertex's coordinates
            pointIn[0] = t1->triangles[triangle].X[i];
            pointIn[1] = t1->triangles[triangle].Y[i];
            pointIn[2] = t1->triangles[triangle].Z[i];
            pointIn[3] = 1;

            //Transform then divide by w for projection
            TransformPoint(result, pointIn, pointOut);
            pointOut[0] = pointOut[0] / pointOut[3];
            pointOut[1] = pointOut[1] / pointOut[3];
            pointOut[2] = pointOut[2] / pointOut[3];
            pointOut[3] = pointOut[3] / pointOut[3];
            
            //Fill in the triangle coordinates
            tempT.X[i] = pointOut[0];
            tempT.Y[i] = pointOut[1];
            tempT.Z[i] = pointOut[2];
            
            //Fill in the triangle colors
            tempT.color[i][0] = t1->triangles[triangle].color[i][0];
            tempT.color[i][1] = t1->triangles[triangle].color[i][1];
            tempT.color[i][2] = t1->triangles[triangle].color[i][2];

        }
        rasterize(&(tempT), canvas, triangle);
    }
}

int vertexLocator(Triangle *triangle, int workOnTop, int *maxVertex, int *leftVertex, int *rightVertex){
    //printf("Vertex Locator\n");
    if(workOnTop){
        //If there is a max vertex available with distinct Y value,
        if(triangle->Y[0] > triangle->Y[1] && triangle->Y[0] > triangle->Y[2]){
            *maxVertex = 0;
            *leftVertex = 1;
            *rightVertex = 2;
        }
        else if(triangle->Y[1] > triangle->Y[0] && triangle->Y[1] > triangle->Y[2]){
            *maxVertex = 1;
            *leftVertex = 0;
            *rightVertex = 2;
        }
        else if(triangle->Y[2] > triangle->Y[1] && triangle->Y[2] > triangle->Y[0]){
            *maxVertex = 2;
            *leftVertex = 0;
            *rightVertex = 1;
        }
        else{
            //printf("ERROR> VertexLocator: No maxVertex found on VertexLocater (workOnTop)\n");
            //printTriangle(triangle);
            return 0;
        }
    }

    else{
        if(triangle->Y[0] < triangle->Y[1] && triangle->Y[0] < triangle->Y[2]){
            *maxVertex = 0;
            *leftVertex = 1;
            *rightVertex = 2;
        }
        else if(triangle->Y[1] < triangle->Y[0] && triangle->Y[1] < triangle->Y[2]){
            *maxVertex = 1;
            *leftVertex = 0;
            *rightVertex = 2;
        }
        else if(triangle->Y[2] < triangle->Y[1] && triangle->Y[2] < triangle->Y[0]){
            *maxVertex = 2;
            *leftVertex = 0;
            *rightVertex = 1;
        }
        else{
            //printf("ERROR> VertexLocator: No maxVertex found on VertexLocater (workOnBottom)\n");
            //printTriangle(triangle);
            return 0;
        }
    }

    //Adjust left and right vertices
    if(triangle->X[*leftVertex] > triangle->X[*rightVertex])
        swapInt(leftVertex, rightVertex);

    return 1;
}

void rasterize(Triangle* triangle, Image *canvas, int triangleid){ 

    /* ERROR HANDLING */
    if(errorFlatTriangleCheck(triangle))
        return;

    /* FIRST WORK ON TOP, THEN BOTTOM*/
    for(int i = 0; i < 2; i++){
        int workOnTop = (i==0 ? 1 : 0);

        int maxVertex = -1;
        int leftVertex = -1;
        int rightVertex = -1;
        

        /* GET THE VERTICES */
        if(workOnTop){
            if(!vertexLocator(triangle, workOnTop, &maxVertex, &leftVertex, &rightVertex))
                continue;
        }
        else{
            if(!vertexLocator(triangle, workOnTop, &maxVertex, &leftVertex, &rightVertex))
                continue;
        }

        /* CALCULATE Y = MX + B VALUES FOR LEFT AND RIGHT EDGES */
        double mL = 0;
        double mR = 0;
        double bL = 0;
        double bR = 0;
        int leftVerticalEdge = 0;
        double leftIntercept = -1;
        int rightVerticalEdge = 0;
        double rightIntercept = -1;
        //If maxVertex is not vertical to left vertex
        if(triangle->X[maxVertex] != triangle->X[leftVertex]){
            mL = (triangle->Y[maxVertex] - triangle->Y[leftVertex])/(triangle->X[maxVertex] - triangle->X[leftVertex]);
            bL = triangle->Y[maxVertex] - mL * triangle->X[maxVertex];
        }
        else if(triangle->X[maxVertex] == triangle->X[leftVertex] && triangle->Y[maxVertex] != triangle->Y[leftVertex]){
            leftVerticalEdge = 1;
            leftIntercept = triangle->X[leftVertex];
        }
        //If maxVertex is not vertical to right vertex
        if(triangle->X[maxVertex] != triangle->X[rightVertex]){
            mR = (triangle->Y[maxVertex] - triangle->Y[rightVertex])/(triangle->X[maxVertex] - triangle->X[rightVertex]);
            bR = triangle->Y[maxVertex] - mR * triangle->X[maxVertex];
        }
        //If maxVertex is vertical to right vertex
        else if(triangle->X[maxVertex] == triangle->X[rightVertex] && triangle->Y[maxVertex] != triangle->Y[rightVertex]){
            rightVerticalEdge = 1;
            rightIntercept = triangle->X[rightVertex];
        }
        
        /* CALCULATE THE SCANLINE RANGE */
        double maxScanline = triangle->Y[maxVertex];
        double minScanline = -1;
        
        if(workOnTop){
        //Right vertex is the middle 
            if(triangle->Y[rightVertex] > triangle->Y[leftVertex])
                minScanline = triangle->Y[rightVertex];
            //Left is the middle
            else if(triangle->Y[leftVertex] > triangle->Y[rightVertex])
                minScanline = triangle->Y[leftVertex];
            //There is a horizontal edge
            else
                minScanline = triangle->Y[leftVertex]; 
        }
        else{
            //left is the middle
            if(triangle->Y[rightVertex] > triangle->Y[leftVertex])
                minScanline = triangle->Y[leftVertex];
            //Right is the middle
            else if(triangle->Y[leftVertex] > triangle->Y[rightVertex])
                minScanline = triangle->Y[rightVertex];
            //There is a horizontal edge
            else
                minScanline = triangle->Y[rightVertex];

            //Adjust the scanline
            swapDouble(&minScanline, &maxScanline);
        }
        
        //Make sure scanline is in the image boundaries
        minScanline = fmax(minScanline, 0);
        maxScanline = fmin(maxScanline, canvas->height - 1);
        
        /* CALCULATE THE LEFT INTERCEPT AND RIGHT INTERCEPT PER SCANLINE, THEN COLOR IT*/
        double leftPoint;
        double rightPoint;
        
        for(int row = C441(minScanline); row <= F441(maxScanline); row++){
            if(leftVerticalEdge)
                leftPoint = leftIntercept;
            else
                leftPoint = (row-bL)/mL;

            if(rightVerticalEdge)
                rightPoint = rightIntercept;
            else
                rightPoint = (row-bR)/mR;

            //Need the original values for color interpolation before swapping
            double ogleft = leftPoint;
            double ogright = rightPoint;

            if(leftPoint > rightPoint)
                swapDouble(&leftPoint, &rightPoint);

            leftPoint = fmax(leftPoint, 0);
            //rightPoint = fmin(rightPoint, WIDTH - 1);
            
            //Z-Buffer
            double zleftPoint = interpolateScalar(row, triangle->Y[leftVertex], triangle->Z[leftVertex], triangle->Y[maxVertex], triangle->Z[maxVertex]);
            double zrightPoint = interpolateScalar(row, triangle->Y[rightVertex], triangle->Z[rightVertex], triangle->Y[maxVertex], triangle->Z[maxVertex]);

            //shading
            double shadingLeft = interpolateScalar(row, triangle->Y[leftVertex], triangle->shading[leftVertex], triangle->Y[maxVertex], triangle->shading[maxVertex]);
            double shadingRight = interpolateScalar(row, triangle->Y[rightVertex], triangle->shading[rightVertex], triangle->Y[maxVertex], triangle->shading[maxVertex]);

            //Color interpolation
            double colorleftR = interpolateScalar(row, triangle->Y[leftVertex], triangle->color[leftVertex][0], triangle->Y[maxVertex], triangle->color[maxVertex][0]);
            double colorrightR = interpolateScalar(row, triangle->Y[rightVertex], triangle->color[rightVertex][0], triangle->Y[maxVertex], triangle->color[maxVertex][0]);

            double colorleftG = interpolateScalar(row, triangle->Y[leftVertex], triangle->color[leftVertex][1], triangle->Y[maxVertex], triangle->color[maxVertex][1]);
            double colorrightG = interpolateScalar(row, triangle->Y[rightVertex], triangle->color[rightVertex][1], triangle->Y[maxVertex], triangle->color[maxVertex][1]);

            double colorleftB = interpolateScalar(row, triangle->Y[leftVertex], triangle->color[leftVertex][2], triangle->Y[maxVertex], triangle->color[maxVertex][2]);
            double colorrightB = interpolateScalar(row, triangle->Y[rightVertex], triangle->color[rightVertex][2], triangle->Y[maxVertex], triangle->color[maxVertex][2]);

            for(int col = C441(leftPoint); col <= F441(rightPoint); col++){
                //Finding Z-point
                double mZ2 = (col - leftPoint)/(rightPoint - leftPoint);
                double zPoint = zleftPoint + mZ2 * (zrightPoint - zleftPoint);
                
                //Interpolating colors 
                double colorM = (col - ogleft)/(ogright - ogleft);
                double R = colorleftR + colorM * (colorrightR - colorleftR);
                double G = colorleftG + colorM * (colorrightG - colorleftG);
                double B = colorleftB + colorM * (colorrightB - colorleftB);

                double shadingM = (col - ogleft)/(ogright - ogleft);
                double shading = shadingLeft + shadingM * (shadingRight - shadingLeft);

                if(zPoint > canvas->depthBuffer[row][col]){
                    canvas->depthBuffer[row][col] = zPoint;
                    if(col<canvas->width)
                        assignPixels(canvas, col, col, row, row, C441(255*fmin(1, R*shading)), C441(fmin(1, G*shading)*255), C441(fmin(1, B*shading)*255));                       
                }
            }
        }
    }
    return;
}

/*
    Writes the color to the corresponding position in the pnm file
*/
void assignPixels(Image *canvas, int colMin, int colMax, int rowMin, int rowMax, unsigned char R, unsigned char G, unsigned char B){

    for (int i = colMin; i <= colMax; i++) {            
        int newRow;
        for(int j = rowMin; j <= rowMax; j++)
            newRow = canvas->height - j - 1;
            if(newRow >= 0){
                int index = (newRow * canvas->width + i) * 3;
                canvas->data[index] = R;
                canvas->data[index + 1] = G;
                canvas->data[index + 2] = B;
            }   
    }
    return;
}

/*
    Writes the image data from the image structure to the FILE descriptor.
*/
void saveImage(Image *canvas, int frame){

    // Open a new file according to the frame number in "write binary" mode
    char str[80];
    sprintf(str, "rasterizer_frame%04d.pnm", frame);
    FILE* fp = fopen(str, "wb");

    // Write the PNM file header
    fprintf(fp, "P6\n%d %d\n255\n", canvas->width, canvas->height);

    //Write image to the file
    fwrite(canvas->data, sizeof(unsigned char), canvas->height * canvas->width * 3, fp);

    fclose(fp);   
    return;
}

void set_initial_depth(Image *canvas){
    for(int row = 0; row < canvas->height; row++){
        for(int col = 0; col < canvas->width; col++){
           canvas->depthBuffer[row][col] = -1;     
        }
    }
}

void PrintMatrix(Matrix m)
{
    for (int i = 0 ; i < 4 ; i++)
    {
        printf("(%.7f %.7f %.7f %.7f)\n", m.A[i][0], m.A[i][1], m.A[i][2], m.A[i][3]);
    }
}

Matrix ComposeMatrices(Matrix M1, Matrix M2)
{
    Matrix m_out;
    for (int i = 0 ; i < 4 ; i++)
        for (int j = 0 ; j < 4 ; j++)
        {
            m_out.A[i][j] = 0;
            for (int k = 0 ; k < 4 ; k++)
                m_out.A[i][j] += M1.A[i][k]*M2.A[k][j];
        }
    return m_out;
}

void  TransformPoint(Matrix m, const double *ptIn, double *ptOut)
{  
    ptOut[0] = ptIn[0]*m.A[0][0]
             + ptIn[1]*m.A[1][0]
             + ptIn[2]*m.A[2][0]
             + ptIn[3]*m.A[3][0];
    ptOut[1] = ptIn[0]*m.A[0][1]
             + ptIn[1]*m.A[1][1]
             + ptIn[2]*m.A[2][1]
             + ptIn[3]*m.A[3][1];
    ptOut[2] = ptIn[0]*m.A[0][2]
             + ptIn[1]*m.A[1][2]
             + ptIn[2]*m.A[2][2]
             + ptIn[3]*m.A[3][2];
    ptOut[3] = ptIn[0]*m.A[0][3]
             + ptIn[1]*m.A[1][3]
             + ptIn[2]*m.A[2][3]
             + ptIn[3]*m.A[3][3];
}

double SineParameterize(int curFrame, int nFrames, int ramp)
{  
    int nNonRamp = nFrames-2*ramp;
    double height = 1./(nNonRamp + 4*ramp/M_PI);
    if (curFrame < ramp)
    {
        double factor = 2*height*ramp/M_PI;
        double eval = cos(M_PI/2*((double)curFrame)/ramp);
        return (1.-eval)*factor;
    }
    else if (curFrame > nFrames-ramp)
    {        
        int amount_left = nFrames-curFrame;
        double factor = 2*height*ramp/M_PI;
        double eval =cos(M_PI/2*((double)amount_left/ramp));
        return 1. - (1-eval)*factor;
    }        
    double amount_in_quad = ((double)curFrame-ramp);
    double quad_part = amount_in_quad*height;
    double curve_part = height*(2*ramp)/M_PI;
    return quad_part+curve_part;
} 

Camera GetCamera(int frame, int nframes)
{            
    double t = SineParameterize(frame, nframes, nframes/10);
    Camera c;
    c.near = 5;
    c.far = 200;
    c.angle = M_PI/6;
    c.position[0] = 40*sin(2*M_PI*t);
    c.position[1] = 40*cos(2*M_PI*t);
    c.position[2] = 40;
    c.focus[0] = 0; 
    c.focus[1] = 0; 
    c.focus[2] = 0;
    c.up[0] = 0;    
    c.up[1] = 1;    
    c.up[2] = 0;    
    return c;       
}

Matrix GetViewTransform(Camera c)
{
    Matrix m;

    //Initialize the matrix with zeros
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++)
            m.A[i][j] = 0;
    }

    //Fill in the values
    m.A[0][0] = 1/tan(c.angle/2);
    m.A[1][1] = 1/tan(c.angle/2);
    m.A[2][2] = (c.far + c.near) / (c.far - c.near);
    m.A[2][3] = -1;
    m.A[3][2] = (2 * c.far * c.near) / (c.far - c.near);
    
    //printf("\nView Transform \n");
    //PrintMatrix(m);
    return m;
}

void subtract_arrays(double* A, double* B, double* result, int size) {
  for (int i = 0; i < size; i++) {
    result[i] = A[i] - B[i];
  }
}

void cross_product(const double A[3], const double B[3], double result[3]) {
    result[0] = A[1]*B[2] - A[2]*B[1]; //A.y*B.z - A.z*B.y
    result[1] = B[0]*A[2] - A[0]*B[2]; //B.x*A.z - A.x*B.z
    result[2] = A[0]*B[1] - A[1]*B[0]; //A.x*B.y - A.y*B.x
}

void set_equal(double* A, double* B, int size) {
  for (int i = 0; i < size; i++) {
    A[i] = B[i];
  }
}




void printCameraFrame(double* U, double* V, double* W, double* O){
    printf("\n");
    printf("Camera Frame: U = %f, %f, %f\n", U[0], U[1], U[2]);
    printf("Camera Frame: V = %f, %f, %f\n", V[0], V[1], V[2]); 
    printf("Camera Frame: W = %f, %f, %f\n", W[0], W[1], W[2]); 
    printf("Camera Frame: O = %f, %f, %f\n", O[0], O[1], O[2]);  
}

Matrix GetCameraTransform(Camera c)
{   
    Matrix rv;
    
    /* CALCULATE THE CAMERA FRAME */
    double T[3];
    double O[3];
    double W[3];
    double V[3];
    double U[3];
    
    //O = camera pos
    set_equal(O, c.position, 3);
    
    //W = O - focus
    subtract_arrays(O, c.focus, W, 3);
    double magnitude = sqrt(W[0]*W[0] + W[1]*W[1] + W[2]*W[2]);
    divideByMagnitude(W, magnitude, 3);
    
    //V = up (temporary solution)
    set_equal(V, c.up, 3);
    double magnitude2 = sqrt(V[0]*V[0] + V[1]*V[1] + V[2]*V[2]);
    divideByMagnitude(V, magnitude2, 3);

    //U = up(V) * (O-focus)(W) 
    cross_product(V, W, U);
    double magnitude3 = sqrt(U[0]*U[0] + U[1]*U[1] + U[2]*U[2]);
    divideByMagnitude(U, magnitude3, 3); 
    
    //V = (O - focus) * U (permanent solution, to make sure V is perp to W)
    cross_product(W, U, V);
    double magnitude4 = sqrt(V[0]*V[0] + V[1]*V[1] + V[2]*V[2]);
    divideByMagnitude(V, magnitude4, 3);

    //T = (0, 0, 0) - O
    double cartesian[3] = {0,0,0};
    subtract_arrays(cartesian, O, T, 3);

    /* CALCULATE THE CAMERA TRANSFORM */
    
    //Initialize with zeros
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++)
            rv.A[i][j] = 0;
    }

    //Fill in the values
    rv.A[0][0] = U[0];
    rv.A[0][1] = V[0];
    rv.A[0][2] = W[0];
    rv.A[1][0] = U[1];
    rv.A[1][1] = V[1];
    rv.A[1][2] = W[1];
    rv.A[2][0] = U[2];
    rv.A[2][1] = V[2];
    rv.A[2][2] = W[2];
    rv.A[3][0] = dot_product(U, T, 3);
    rv.A[3][1] = dot_product(V, T, 3);
    rv.A[3][2] = dot_product(W, T, 3);
    rv.A[3][3] = 1;
    
    return rv;
}

Matrix GetDeviceTransform(Camera c)
{   
    Matrix rv;
    double n = 1000;
    double m = 1000;

    /* YOU IMPLEMENT THIS */
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            rv.A[i][j] = 0;
        }
    }
    rv.A[0][0] = n/2;
    rv.A[1][1] = m/2;
    rv.A[2][2] = 1;
    rv.A[3][0] = n/2;
    rv.A[3][1] = m/2;
    rv.A[3][3] = 1;

    return rv;
}

char *Read3Numbers(char *tmp, double *v1, double *v2, double *v3)
{
    *v1 = atof(tmp);
    while (*tmp != ' ')
       tmp++;
    tmp++; /* space */
    *v2 = atof(tmp);
    while (*tmp != ' ')
       tmp++;
    tmp++; /* space */
    *v3 = atof(tmp);
    while (*tmp != ' ' && *tmp != '\n')
       tmp++;
    return tmp;
}

TriangleList *Get3DTriangles()
{
   FILE *f = fopen("ws_tris.txt", "r");
   if (f == NULL)
   {
       fprintf(stderr, "You must place the ws_tris.txt file in the current directory.\n");
       exit(EXIT_FAILURE);
   }
   fseek(f, 0, SEEK_END);
   int numBytes = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (numBytes != 3892295)
   {
       fprintf(stderr, "Your ws_tris.txt file is corrupted.  It should be 3892295 bytes, but you have %d.\n", numBytes);
       exit(EXIT_FAILURE);
   }

   char *buffer = (char *) malloc(numBytes);
   if (buffer == NULL)
   {
       fprintf(stderr, "Unable to allocate enough memory to load file.\n");
       exit(EXIT_FAILURE);
   }
   
   fread(buffer, sizeof(char), numBytes, f);

   char *tmp = buffer;
   int numTriangles = atoi(tmp);
   while (*tmp != '\n')
       tmp++;
   tmp++;
 
   if (numTriangles != 14702)
   {
       fprintf(stderr, "Issue with reading file -- can't establish number of triangles.\n");
       exit(EXIT_FAILURE);
   }

   TriangleList *tl = (TriangleList *) malloc(sizeof(TriangleList));
   tl->numTriangles = numTriangles;
   tl->triangles = (Triangle *) malloc(sizeof(Triangle)*tl->numTriangles);

   for (int i = 0 ; i < tl->numTriangles ; i++)
   {
       for (int j = 0 ; j < 3 ; j++)
       {
           double x, y, z;
           double r, g, b;
           double normals[3];
/*
 * Weird: sscanf has a terrible implementation for large strings.
 * When I did the code below, it did not finish after 45 minutes.
 * Reading up on the topic, it sounds like it is a known issue that
 * sscanf fails here.  Stunningly, fscanf would have been faster.
 *     sscanf(tmp, "(%lf, %lf), (%lf, %lf), (%lf, %lf) = (%d, %d, %d)\n%n",
 *              &x1, &y1, &x2, &y2, &x3, &y3, &r, &g, &b, &numRead);
 *
 *  So, instead, do it all with atof/atoi and advancing through the buffer manually...
 */
           tmp = Read3Numbers(tmp, &x, &y, &z);
           tmp += 3; /* space+slash+space */
           tmp = Read3Numbers(tmp, &r, &g, &b);
           tmp += 3; /* space+slash+space */
           tmp = Read3Numbers(tmp, normals+0, normals+1, normals+2);
           tmp++;    /* newline */

           tl->triangles[i].X[j] = x;
           tl->triangles[i].Y[j] = y;
           tl->triangles[i].Z[j] = z;
           tl->triangles[i].color[j][0] = r;
           tl->triangles[i].color[j][1] = g;
           tl->triangles[i].color[j][2] = b;
#ifdef NORMALS
           tl->triangles[i].normals[j][0] = normals[0];
           tl->triangles[i].normals[j][1] = normals[1];
           tl->triangles[i].normals[j][2] = normals[2];
#endif
       }
   }

   free(buffer);
   return tl;
}

double C441(double f)
{
    return ceil(f-0.00001);
}

double F441(double f)
{
    return floor(f+0.00001);
}

int errorFlatTriangleCheck(Triangle* triangle){



    if(triangle->Y[0] == triangle->Y[1] && triangle->Y[1] == triangle->Y[2]){
        //printf("ERROR> Invalid triangle: All the Y values are the same\n");
        //printTriangle(triangle);
        return 1;
    }
    if(triangle->X[0] == triangle->X[1] && triangle->X[1] == triangle->X[2]){
        //printf("ERROR> Invalid triangle: All the X values are the same\n");
        //printTriangle(triangle);
        return 1;
    }
    if(triangle->X[0] == triangle->X[1] && triangle->Y[0] == triangle->Y[1]){
        //printf("ERROR> Invalid triangle: Vertices 0 and 1 are the same\n");
        //printTriangle(triangle);
        return 1;
    }
    else if(triangle->X[1] == triangle->X[2] && triangle->Y[1] == triangle->Y[2]){
        //printf("ERROR> Invalid triangle: Vertices 1 and 2 are the same\n");
        //printTriangle(triangle);
        return 1;
    }
    else if(triangle->X[0] == triangle->X[2] && triangle->Y[0] == triangle->Y[2]){
        //printf("ERROR> Invalid triangle: Vertices 0 and 2 are the same\n");
        //printTriangle(triangle);
        return 1;
    }
   
    return 0;
}

double interpolateScalar(double X, double A, double fA, double B, double fB){

    if(A == B)
        return fA;
    
    double t = (X - A) / (B - A);
    double Y = fA + t * (fB - fA);

    return Y;
}

void swapDouble(double *a, double *b){
    double temp = *a;
    *a = *b;
    *b = temp;
    return;
}

void swapInt(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
    return;
}

void printTriangle(Triangle* triangle){
    printf("(%f, %f, %f) / (%f, %f, %f), (%f, %f, %f) / (%f, %f, %f), (%f, %f, %f) / (%f, %f, %f)\n", triangle->X[0], triangle->Y[0], triangle->Z[0], triangle->color[0][0], triangle->color[0][1], triangle->color[0][2], triangle->X[1], triangle->Y[1], triangle->Z[1], triangle->color[1][0], triangle->color[1][1], triangle->color[1][2], triangle->X[2], triangle->Y[2], triangle->Z[2], triangle->color[2][0], triangle->color[2][1], triangle->color[2][2]);
    return;
}