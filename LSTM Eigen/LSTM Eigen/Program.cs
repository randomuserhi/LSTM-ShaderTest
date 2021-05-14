using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Threading;

namespace LSTM_Eigen
{
    class Program
    {
        static LSTMEvol[] Population = new LSTMEvol[500];

        static Queue<(int, int)> Jobs = new Queue<(int, int)>();
        static object Lock = new object();
        static int Co = 0;

        static void Main()
        {
            Process process = Process.GetCurrentProcess();
            int offset = process.Threads.Count;
            int cpuCount = Environment.ProcessorCount;
            Console.WriteLine("CPU Count : {0}", cpuCount);

            /*for (int j = 0; j < 1; j++)
            {
                LSTMEvol A = new LSTMEvol(new int[] { 2, 10, 10, 10, 10, 2 });
                for (int z = 0; z < 100; z++)
                {
                    float[] O = A.FeedForward(new float[] { 1, 1 });
                    for (int i = 0; i < O.Length; i++)
                    {
                        Console.Write(O[i] + ", ");
                    }
                    Console.WriteLine("");
                }
            }
            Console.ReadLine();*/

            for (int i = 0; i < Population.Length; i++)
            {
                Population[i] = new LSTMEvol(new int[] { 2, 2 });
            }

            for (int i = 0; i < 1000; i++)
            {
                float[] R = Population[0].FeedForward(new float[] { 1, 1 });
                Console.WriteLine(String.Join(",", R));
            }
            Console.ReadLine();

            /*Stopwatch S = new Stopwatch();
            S.Start();
            for (int i = 0; i < Population.Length; i++)
            {
                Population[i].FeedForward(new float[] { 1, 1 });
            }
            S.Stop();

            Console.WriteLine(S.ElapsedMilliseconds + "ms");
            Console.ReadLine();*/
            
            
            /*Thread A = new Thread(() => {
                while (true)
                {
                    (int, int) J;
                    lock(Lock)
                    {
                        while (Jobs.Count == 0) Monitor.Wait(Lock);
                        J = Jobs.Dequeue();
                    }
                    Func(J.Item1, J.Item2);
                    return;
                }
            });

            Thread B = new Thread(() => {
                while (true)
                {
                    (int, int) J;
                    lock (Lock)
                    {
                        while (Jobs.Count == 0) Monitor.Wait(Lock);
                        J = Jobs.Dequeue();
                    }
                    Func(J.Item1, J.Item2);
                    return;
                }
            });

            Thread C = new Thread(() => {
                while (true)
                {
                    (int, int) J;
                    lock (Lock)
                    {
                        while (Jobs.Count == 0) Monitor.Wait(Lock);
                        J = Jobs.Dequeue();
                    }
                    Func(J.Item1, J.Item2);
                    return;
                }
            });

            Thread D = new Thread(() => {
                while (true)
                {
                    (int, int) J;
                    lock (Lock)
                    {
                        while (Jobs.Count == 0) Monitor.Wait(Lock);
                        J = Jobs.Dequeue();
                    }
                    Func(J.Item1, J.Item2);
                    return;
                }
            });

            Thread E = new Thread(() => {
                while (true)
                {
                    (int, int) J;
                    lock (Lock)
                    {
                        while (Jobs.Count == 0) Monitor.Wait(Lock);
                        J = Jobs.Dequeue();
                    }
                    Func(J.Item1, J.Item2);
                    return;
                }
            });

            Thread F = new Thread(() => {
                while (true)
                {
                    (int, int) J;
                    lock (Lock)
                    {
                        while (Jobs.Count == 0) Monitor.Wait(Lock);
                        J = Jobs.Dequeue();
                    }
                    Func(J.Item1, J.Item2);
                    return;
                }
            });

            Thread G = new Thread(() => {
                while (true)
                {
                    (int, int) J;
                    lock (Lock)
                    {
                        while (Jobs.Count == 0) Monitor.Wait(Lock);
                        J = Jobs.Dequeue();
                    }
                    Func(J.Item1, J.Item2);
                    return;
                }
            });

            Thread H = new Thread(() => {
                while (true)
                {
                    (int, int) J;
                    lock (Lock)
                    {
                        while (Jobs.Count == 0) Monitor.Wait(Lock);
                        J = Jobs.Dequeue();
                    }
                    Func(J.Item1, J.Item2);
                    return;
                }
            });

            A.Priority = ThreadPriority.Highest;
            A.IsBackground = true;
            B.Priority = ThreadPriority.Highest;
            B.IsBackground = true;
            C.Priority = ThreadPriority.Highest;
            C.IsBackground = true;
            D.Priority = ThreadPriority.Highest;
            D.IsBackground = true;
            E.Priority = ThreadPriority.Highest;
            E.IsBackground = true;
            F.Priority = ThreadPriority.Highest;
            F.IsBackground = true;
            G.Priority = ThreadPriority.Highest;
            G.IsBackground = true;
            H.Priority = ThreadPriority.Highest;
            H.IsBackground = true;

            A.Start();
            B.Start();
            C.Start();
            D.Start();
            E.Start();
            F.Start();
            G.Start();
            H.Start();

            process.Refresh();
            for (int i = 0; i < 8; ++i)
            {
                //process.Threads[i + offset].ProcessorAffinity = (IntPtr)(1L << i);
                //The code above distributes threads evenly on all processors.
                //But now we are making a test, so let's bind all the threads to the
                //second processor.
                process.Threads[i + offset].ProcessorAffinity = (IntPtr)(1L << i);
            }

            S = new Stopwatch();

            lock (Lock)
            {
                Jobs.Enqueue((0, 250));
                Jobs.Enqueue((250, 500));
                Jobs.Enqueue((500, 750));
                Jobs.Enqueue((750, 1000));
                Jobs.Enqueue((1000, 1250));
                Jobs.Enqueue((1250, 1500));
                Jobs.Enqueue((1500, 1750));
                Jobs.Enqueue((1750, 2000));
                Monitor.PulseAll(Lock);
            }

            S.Start();

            A.Join();
            B.Join();
            C.Join();
            D.Join();
            E.Join();
            F.Join();
            G.Join();
            H.Join();

            S.Stop();

            Console.WriteLine(S.ElapsedMilliseconds + "ms");
            Console.ReadLine();

            Console.WriteLine(Jobs.Count);

            A.Abort();
            B.Abort();
            C.Abort();
            D.Abort();
            E.Abort();
            F.Abort();
            G.Abort();
            H.Abort();*/
        }

        /*static void Func(int Start, int End)
        {
            for (int i = Start; i < End; i++)
            {
                Population[i].FeedForward(new float[] { 1, 1 });
            }
        }*/
    }
}
