/*
Время отправки сообщения рассчитывается по формуле Tn = Ts + Tb*N.
Будем считать, что размер int равен 4 байтам.
  - 0 процесс выполнит критическую секцию бесконечно быстро
  - следующие процессы разошлют по 24 сообщения REQUEST (4 байта) + получат один ответ с токеном (204 байт)
  - для синхронизации счётчика каждый процесс посылает 24 сообщения (4 байт) 
Тогда T = 24*(24*(100+1*4) + 100 + 204*1) + 25*24*(100 + 4*1) = 189600
*/

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <ctime>
#include <unistd.h>
#include <filesystem>
#include <fstream>

using namespace std;

int main(int argc, char *argv[]) {
    std::filesystem::path filePath = "critical.txt";
    int rank, size, root = 0;
    MPI_Datatype tokenType;
    MPI_Status status;

    int i, j, probe_flag, t, t1;
    int reqTag = 0, tokenTag = 1, counterTag = 2;
    float inc1, inc2 = 0.05;
    enum states_enum { RELEASED, REQUESTED, GRANTED } state;
    bool hasToken, running, waiting, found;
    state = RELEASED;
    int localProcessesInCS = 0;
    int processesInCS = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Type_contiguous(51, MPI_INT, &tokenType);
    MPI_Type_commit(&tokenType);

    int RN[size], SN;
    struct token_struct { int LN[25]; int Q[25]; int Qsize; } token;
    token.Qsize = 0;

    for (i = 0; i < size; i++) {
        token.LN[i] = 0;
        RN[i] = 0;
    }

    if (rank == root) {
        hasToken = false;
        t = rand() % size;
        for (i = 0; i < size; i++) {
            if (i != root) {
                if (i == t)
                    t1 = true;
                else
                    t1 = false;
                MPI_Send(&t1, 1, MPI_INT, i, tokenTag, MPI_COMM_WORLD);
            } else if (t == root) {
                hasToken = true;
                cout << "Process " << rank << ": starts with token\n";
            }
        }
    } else {
        MPI_Recv(&hasToken, 1, MPI_INT, root, tokenTag, MPI_COMM_WORLD, &status);
        if (hasToken)
            cout << "Process " << rank << ": starts with token\n";
    }

    running = true;
    MPI_Barrier(MPI_COMM_WORLD);

    while (running) {
        if (state == RELEASED) {
            waiting = true;

            while (waiting) {
                if (localProcessesInCS == 0) {
                  waiting = false;
                  break;
                }
                if (MPI_Wtime() >= inc1) { //чтобы проверять с некоторым интервалом
                    MPI_Iprobe(MPI_ANY_SOURCE, reqTag, MPI_COMM_WORLD, &probe_flag, &status);

                    if (probe_flag) {
                        MPI_Recv(&SN, 1, MPI_INT, MPI_ANY_SOURCE, reqTag, MPI_COMM_WORLD, &status);
                        if (SN > RN[status.MPI_SOURCE])
                            RN[status.MPI_SOURCE] = SN;

                        cout << "Process " << rank << " received REQUEST from process " << status.MPI_SOURCE << endl;
                        if (hasToken && (RN[status.MPI_SOURCE] == (token.LN[status.MPI_SOURCE] + 1))) {
                            cout << "Process " << rank << " is sending the token to process " << status.MPI_SOURCE << "\n";
                            MPI_Send(&token, 1, tokenType, status.MPI_SOURCE, tokenTag, MPI_COMM_WORLD);
                            hasToken = false;
                        }
                    }

                    MPI_Iprobe(MPI_ANY_SOURCE, counterTag, MPI_COMM_WORLD, &probe_flag, &status);
                    if (probe_flag)
                        MPI_Recv(&processesInCS, 1, MPI_INT, MPI_ANY_SOURCE, counterTag, MPI_COMM_WORLD, &status);

                    inc1 = MPI_Wtime() + inc2;
                }

                //MPI_Allreduce(&localProcessesInCS, &processesInCS, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

                if (processesInCS == size) {
                    waiting = false;
                }
            }

            if (processesInCS == size) {
              break;
            }

            if (hasToken)
                state = GRANTED;
            else {
                RN[rank]++;
                cout << "Process " << rank << " number " << RN[rank] << " is sending REQUEST\n";
                cout << "message (" << rank << ":" << RN[rank] << ")\n";
                for (i = 0; i < size; i++) { // не используем MPI_Bcast, т.к. нельзя указать тег
                    if (i != rank)
                        MPI_Send(&RN[rank], 1, MPI_INT, i, reqTag, MPI_COMM_WORLD);
                    }

                state = REQUESTED;
                inc1 = 0;
            }
        }

        if (state == REQUESTED) {
          while (!hasToken) {
            MPI_Iprobe(MPI_ANY_SOURCE, reqTag, MPI_COMM_WORLD, &probe_flag, &status);

            if (probe_flag) {
                MPI_Recv(&SN, 1, MPI_INT, MPI_ANY_SOURCE, reqTag, MPI_COMM_WORLD, &status);
                if (SN > RN[status.MPI_SOURCE])
                    RN[status.MPI_SOURCE] = SN;

                cout << "Process " << rank << " received REQUEST from process " << status.MPI_SOURCE << "\n";
            }

            MPI_Iprobe(MPI_ANY_SOURCE, tokenTag, MPI_COMM_WORLD, &probe_flag, &status);

            if (probe_flag) {
                MPI_Recv(&token, 1, tokenType, MPI_ANY_SOURCE, tokenTag, MPI_COMM_WORLD, &status);
                cout << "Process " << rank << " received the token from process " << status.MPI_SOURCE << endl;
                state = GRANTED;
                hasToken = true;
            }

            MPI_Iprobe(MPI_ANY_SOURCE, counterTag, MPI_COMM_WORLD, &probe_flag, &status);
            if (probe_flag) {
                MPI_Recv(&processesInCS, 1, MPI_INT, MPI_ANY_SOURCE, counterTag, MPI_COMM_WORLD, &status);
            }
          }
        }

        if (state == GRANTED) {
            cout << "Process " << rank << " entering critical section\n"; 

            if (std::filesystem::exists(filePath)) {
                cerr << "Error: File 'critical.txt' already exists." << endl;
                return 1;
            }

            std::ofstream file(filePath.c_str());
            if (!file.is_open()) {
                cerr << "Error: Unable to create 'critical.txt'." << endl;
                return 1;
            }

            std::srand(std::time(nullptr));
            int randomTime = std::rand() % 5 + 1;
            sleep(randomTime);

            std::filesystem::remove(filePath);

            
            MPI_Iprobe(MPI_ANY_SOURCE, reqTag, MPI_COMM_WORLD, &probe_flag, &status);
            if (probe_flag) {
                MPI_Recv(&SN, 1, MPI_INT, MPI_ANY_SOURCE, reqTag, MPI_COMM_WORLD, &status);
                if (SN > RN[status.MPI_SOURCE])
                    RN[status.MPI_SOURCE] = SN;
            }

            token.LN[rank] = RN[rank];

            for (i = 0; i < size; i++) {
                if ((i != rank) && (RN[i] == (token.LN[i] + 1))) {
                    found = false;

                    for (j = 0; j < token.Qsize; j++) {
                        if (token.Q[j] == i) {
                            found = true;
                            j = token.Qsize;
                        }
                    }

                    if (!found) {
                        token.Q[token.Qsize] = i;
                        token.Qsize++;
                    }
                }
            }

            cout << "Process " << rank << " has exited critical section\n";
            localProcessesInCS++;
            processesInCS++;

            for (i = 0; i < size; i++) { // не используем MPI_Bcast, т.к. нельзя указать тег
                if (i != rank)
                    MPI_Send(&processesInCS, 1, MPI_INT, i, counterTag, MPI_COMM_WORLD);
            }

            if (token.Qsize > 0) {
                t = token.Q[0];
                token.Qsize--;

                for (i = 0; i < token.Qsize; i++)
                    token.Q[i] = token.Q[i + 1];

                cout << "Rank " << rank << " is sending the token to rank " << t << "\n";
                MPI_Send(&token, 1, tokenType, t, tokenTag, MPI_COMM_WORLD);
                hasToken = false;
            }

            state = RELEASED;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Process " << rank << " finished\n";
    MPI_Finalize();
    return 0;
}
