#include "3mm.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>

int size, rank;
MPI_Comm main_comm;
int errcode = 0;

void initialize_matrix(float* matrix, int rows, int cols, int rank, int real) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (float)((rand() + rank*j) % 10 + 1)/10; 
        }
    }
    for (int i = rows*cols; i < real; ++i)
        matrix[i] = 0.0;
}

void print_matrix(float* local_matrix, int local_rows, int cols, const char* title, int rank, int size) {
    // Синхронизация вывода для избежания перемешивания строк при выводе
    MPI_Barrier(main_comm);

    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("Rank %d - %s:\n", rank, title);
            for (int j = 0; j < local_rows; j++) {
                for (int k = 0; k < cols; k++) {
                    printf("%8.3f ", local_matrix[j * cols + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
        // Синхронизация перед переходом к следующему процессу
        MPI_Barrier(main_comm);
    }
}

int exchange_blocks(float* my_block, int my_block_size, int my_rank, int num_procs) {
    int dest = (my_rank + 1) % num_procs; // процесс, которому будем отправлять блок
    int source = (my_rank - 1 + num_procs) % num_procs; // процесс, от которого будем получать блок
    MPI_Status status;
    int sendrecv_error = MPI_Sendrecv_replace(my_block, my_block_size, MPI_FLOAT, dest, 0, source, 0, main_comm, &status);
    if (sendrecv_error)
        return 1;
    return 0;
}


void matrix_multiply_block(float* A, float* B, float* C, int rows, int cols, int commonDim, int ni, int y, int realSize) {
    //rows - в А, cols - в B
    int skip = 0;
    for (int i = 0; i < (size + rank - y) % size; ++i) {
        int blockSize = ni / size + (i < ni % size ? 1 : 0);
        skip += blockSize;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            C[i * ni + skip + j] = 0.0;
            for (int k = 0; k < commonDim; k++) {
                C[i * ni + skip + j] += A[i * commonDim + k] * B[k * cols + j];
            }
        }
    }
}

int multiply_matr(float* A, float* B, float* C, int rows, int commonDim, int realSize, int ni) {
    MPI_Barrier(main_comm);
    int err;
    for (int i = 0; i < size; ++i) {
        int blockSize = ni / size + (((size + rank - i) % size) < ni % size ? 1 : 0);
        matrix_multiply_block(A, B, C, rows, blockSize, commonDim, ni, i, realSize);
        err = exchange_blocks(B, realSize*commonDim, rank, size);
        if (err)
            return 1;
    }
    return 0;
}

void write_matrix_to_file(MPI_File file, float* matrix, int rows, int cols, int offset) {
    MPI_Offset disp = offset * sizeof(float);
    MPI_Barrier(main_comm);
    MPI_File_write_at(file, disp, matrix, rows * cols, MPI_FLOAT, MPI_STATUS_IGNORE);
}

void read_matrix_from_file(MPI_File file, float* matrix, int rows, int cols, int offset) {
    MPI_Offset disp = offset * sizeof(float);
    MPI_Barrier(main_comm);
    MPI_File_read_at(file, disp, matrix, rows * cols, MPI_FLOAT, MPI_STATUS_IGNORE);
}

static void ehandle(MPI_Comm *comm, int *err, ...) {
    int len;
    char errstr[MPI_MAX_ERROR_STRING];
    MPI_Comm_rank(main_comm, &rank);
    MPI_Comm_size(main_comm, &size);
    MPI_Error_string(*err, errstr, &len);
    printf("Rank %d: notified of error %s\n", rank, errstr);
    MPIX_Comm_shrink(main_comm, &main_comm);
    MPI_Comm_rank(main_comm, &rank);
    MPI_Comm_size(main_comm, &size);
    errcode = 1;
}


int main(int argc, char** argv) {
    int ni, nj, nk, nl, nm;
    int suite_size;
    bool first = false;

    if (argc < 2) {
        fprintf(stderr, "Usage: mpicc 3mm.c -o 3mm && mpirun -np 10 --oversubscribe --with-ft ulfm --allow-run-as-root ./3mm 1\n");
        return -1;
    } else {
        suite_size = atoi(argv[1]);
    }
    get_sizes(suite_size, &ni, &nj, &nk, &nl, &nm);

    MPI_Init(&argc, &argv);

    int offset;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 1)
        first = true;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    main_comm = MPI_COMM_WORLD;

    MPI_Errhandler eh;
    MPI_Comm_create_errhandler(ehandle, &eh);
    MPI_Comm_set_errhandler(main_comm, eh);
    MPI_Barrier(main_comm);

    srand(time(NULL));

    // Вычисление размера блока для каждой размерности
    int blockSizeA = ni / size + (rank < ni % size ? 1 : 0);
    int blockSizeB = nj / size + (rank < nj % size ? 1 : 0);
    int realSizeB = nj / size + 1;

    // Выделяем память для блоков матриц A, B, C, D, E, F, G
    float* A = malloc(blockSizeA * nk * sizeof(float));
    float* B = malloc(nk * realSizeB * sizeof(float));
    float* E = malloc(blockSizeA * nj * sizeof(float));

    // Инициализация матриц
    initialize_matrix(A, blockSizeA, nk, rank, blockSizeA*nk);
    initialize_matrix(B, nk, blockSizeB, rank, realSizeB*nk);

    // Вычисление части произведения матрицы
    //void multiply_matr(float* A, float* B, float* C, int rows, int commonDim, int rank, int size, int realSize, int ni);
    checkpoint1:
    if (errcode) {
        blockSizeA = ni / size + (rank < ni % size ? 1 : 0);
        blockSizeB = nj / size + (rank < nj % size ? 1 : 0);
        realSizeB = nj / size + 1;

        // Выделяем память для блоков матриц A, B, C, D, E, F, G
        A = malloc(blockSizeA * nk * sizeof(float));
        B = malloc(nk * realSizeB * sizeof(float));
        E = malloc(blockSizeA * nj * sizeof(float));

        // Инициализация матриц
        initialize_matrix(A, blockSizeA, nk, rank, blockSizeA*nk);
        initialize_matrix(B, nk, blockSizeB, rank, realSizeB*nk);
        errcode = 0;
    }

    print_matrix(A, blockSizeA, nk, "Matrix A", rank, size);
    fflush(stdout);
    print_matrix(B, nk, blockSizeB, "Matrix B", rank, size);
    fflush(stdout);

    multiply_matr(A, B, E, blockSizeA, nk, realSizeB, nj);
    if (errcode) {
        free(A);
        free(B);
        free(E);
        goto checkpoint1;
    }

    print_matrix(E, blockSizeA, nj, "Matrix E (Result)", rank, size);
    fflush(stdout);

    MPI_File file;
    MPI_File_delete("result_E.bin", MPI_INFO_NULL);
    MPI_File_open(main_comm, "result_E.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    offset = 0;
    for (int i = 0; i < rank; ++i) {
        offset += ni / size + (i < ni % size ? 1 : 0);
    }
    offset *= nj;
    write_matrix_to_file(file, E, blockSizeA, nj, offset);
    MPI_File_close(&file);

    free(A);
    free(B);
    int blockSizeC = nm / size + (rank < nm % size ? 1 : 0);
    int realSizeC = nm / size + 1;

    float* C = malloc(nj * realSizeC * sizeof(float));
    float* F = malloc(blockSizeA * nm * sizeof(float));
    initialize_matrix(C, nj, blockSizeC, rank, nj*realSizeC);


    /*  Тестирование записи и чтения из файла
    MPI_Barrier(MPI_COMM_WORLD);
    print_matrix(E, blockSizeA, nj, "Matrix E (Result)", rank, size);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "result_E.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    offset = 0;
    for (int i = 0; i < rank; ++i) {
        offset += ni / size + (i < ni % size ? 1 : 0);
    }
    offset *= nj;
    write_matrix_to_file(file, E, blockSizeA, nj, offset);
    MPI_Barrier(MPI_COMM_WORLD);
    print_matrix(E, blockSizeA, nj, "Matrix E (Result)", rank, size);
    for (int i = 0; i < blockSizeA; i++) {
        for (int j = 0; j < nj; j++) {
            E[i * nj + j] = 0.0; 
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    print_matrix(E, blockSizeA, nj, "Matrix E (Result)", rank, size);
    read_matrix_from_file(file, E, blockSizeA, nj, offset);
    MPI_Barrier(MPI_COMM_WORLD);
    print_matrix(E, blockSizeA, nj, "Matrix E (Result)", rank, size);
    MPI_File_close(&file);*/

    checkpoint2:

    if (errcode) {
        blockSizeA = ni / size + (rank < ni % size ? 1 : 0);
        blockSizeC = nm / size + (rank < nm % size ? 1 : 0);
        realSizeC = nm / size + 1;

        C = malloc(nj * realSizeC * sizeof(float));
        E = malloc(blockSizeA * nj * sizeof(float));
        F = malloc(blockSizeA * nm * sizeof(float));

        // Инициализация матриц
        initialize_matrix(C, nj, blockSizeC, rank, nj*realSizeC);
        errcode = 0;
        MPI_File file;
        MPI_File_open(main_comm, "result_E.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        offset = 0;
        for (int i = 0; i < rank; ++i) {
            offset += ni / size + (i < ni % size ? 1 : 0);
        }
        offset *= nj;
        read_matrix_from_file(file, E, blockSizeA, nj, offset);
        MPI_Barrier(main_comm);
        MPI_File_close(&file);
        print_matrix(E, blockSizeA, nj, "Matrix E (Result)", rank, size);
    }

    print_matrix(C, nj, blockSizeC, "Matrix C", rank, size);
    fflush(stdout);
    if ((rank == 1) && (first)) {
        printf("Process %d kills himself\n", rank);
        raise(SIGKILL);
        first = false;
    }

    multiply_matr(E, C, F, blockSizeA, nj, realSizeC, nm);
    if (errcode) {
        free(C);
        free(E);
        free(F);
        goto checkpoint2;
    }

    print_matrix(F, blockSizeA, nm, "Matrix F (Result)", rank, size);
    fflush(stdout);

    MPI_File_delete("result_F.bin", MPI_INFO_NULL);
    MPI_File_open(main_comm, "result_F.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    offset = 0;
    for (int i = 0; i < rank; ++i) {
        offset += ni / size + (i < ni % size ? 1 : 0);
    }
    offset *= nm;
    write_matrix_to_file(file, F, blockSizeA, nm, offset);
    MPI_File_close(&file);

    int blockSizeD = nl / size + (rank < nl % size ? 1 : 0);
    int realSizeD = nl / size + 1;

    free(C);
    free(E);
    float* D = malloc(nm * realSizeD * sizeof(float));
    float* G = malloc(blockSizeA * nl * sizeof(float));

    // Инициализация матриц
    initialize_matrix(D, nm, blockSizeD, rank, nm*realSizeD);

    checkpoint3:

    if (errcode) {
        blockSizeA = ni / size + (rank < ni % size ? 1 : 0);
        blockSizeD = nl / size + (rank < nl % size ? 1 : 0);
        realSizeD = nl / size + 1;

        D = malloc(nm * realSizeD * sizeof(float));
        F = malloc(blockSizeA * nm * sizeof(float));
        G = malloc(blockSizeA * nl * sizeof(float));

        // Инициализация матриц
        initialize_matrix(D, nm, blockSizeD, rank, nm*realSizeD);
        errcode = 0;
        MPI_File file;
        MPI_File_open(main_comm, "result_F.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        offset = 0;
        for (int i = 0; i < rank; ++i) {
            offset += ni / size + (i < ni % size ? 1 : 0);
        }
        offset *= nm;
        read_matrix_from_file(file, F, blockSizeA, nm, offset);
        MPI_Barrier(main_comm);
        MPI_File_close(&file);
    }

    print_matrix(D, nm, blockSizeD, "Matrix D", rank, size);
    fflush(stdout);

    multiply_matr(F, D, G, blockSizeA, nm, realSizeD, nl);
    if (errcode) {
        free(D);
        free(F);
        free(G);
        goto checkpoint3;
    }
    print_matrix(G, blockSizeA, nl, "Matrix G (Result)", rank, size);
    fflush(stdout);

    // Освобождение выделенной памяти
    free(D);
    free(F);
    free(G);

    MPI_Finalize();

    return 0;
}