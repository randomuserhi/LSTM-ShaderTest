using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LSTMManager
{
    public static ComputeShader GenerateComputeShader()
    {
        return Resources.Load<ComputeShader>("LSTM");
    }

    public struct MatrixInfo
    {
        public int Rows;
        public int Cols;
        public int Size;
        public int Offset; //Offset in memory for the buffers
    }

    public struct LSTMGroup
    {
        public int Population;
        public int[] Structure;
        public ComputeBuffer _Structure;

        public float[] Inputs;
        public float[] Nodes;
        public float[] Outputs;
        public ComputeBuffer _Inputs;
        public ComputeBuffer _Outputs;
        public ComputeBuffer _Nodes;

        public float[] CellStates;
        public ComputeBuffer _CellStates;

        public float[] InputWeights;
        public float[] InputStateWeights;
        public float[] InputBias;
        public ComputeBuffer _InputWeights;
        public ComputeBuffer _InputStateWeights;
        public ComputeBuffer _InputBias;

        public float[] InputModulationWeights;
        public float[] InputModulationStateWeights;
        public float[] InputModulationBias;
        public ComputeBuffer _InputModulationWeights;
        public ComputeBuffer _InputModulationStateWeights;
        public ComputeBuffer _InputModulationBias;

        public float[] ForgetWeights;
        public float[] ForgetStateWeights;
        public float[] ForgetBias;
        public ComputeBuffer _ForgetWeights;
        public ComputeBuffer _ForgetStateWeights;
        public ComputeBuffer _ForgetBias;

        public float[] OutputWeights;
        public float[] OutputStateWeights;
        public float[] OutputBias;
        public ComputeBuffer _OutputWeights;
        public ComputeBuffer _OutputStateWeights;
        public ComputeBuffer _OutputBias;

        public MatrixInfo[] NodeCellInfo;
        public MatrixInfo[] WeightBiasInfo;
        public int MaxGateSize;
        public ComputeBuffer _NodeCellInfo;
        public ComputeBuffer _WeightBiasInfo;

        public float[] InputGate;
        public float[] InputModulationGate;
        public float[] ForgetGate;
        public float[] OutputGate;
        public ComputeBuffer _InputGate;
        public ComputeBuffer _InputModulationGate;
        public ComputeBuffer _ForgetGate;
        public ComputeBuffer _OutputGate;
    }

    public static LSTMGroup CreateLSTMGroup(int[] Structure, int Population)
    {
        int NodeCount = 0;
        int WeightCount = 0;
        int StateWeightCount = 0;
        MatrixInfo[] NodeCellInfo = new MatrixInfo[Structure.Length];
        MatrixInfo[] WeightBiasInfo = new MatrixInfo[2 * (Structure.Length - 1)];
        int MaxGateSize = Structure[0];

        NodeCellInfo[0] = new MatrixInfo()
        {
            Rows = 1,
            Cols = Structure[0],
            Size = Structure[0],
            Offset = 0
        };
        for (int i = 1, j = 0; i < Structure.Length; i++, j++)
        {
            WeightCount += Structure[j] * Structure[i];
            StateWeightCount += Structure[i] * Structure[i];
            NodeCount += Structure[i];

            NodeCellInfo[i] = new MatrixInfo()
            {
                Rows = 1,
                Cols = Structure[i],
                Size = Structure[i],
                Offset = 0
            };

            WeightBiasInfo[j * 2] = new MatrixInfo()
            {
                Rows = Structure[j],
                Cols = Structure[i],
                Size = Structure[j] * Structure[i],
                Offset = 0
            };
            WeightBiasInfo[j * 2 + 1] = new MatrixInfo()
            {
                Rows = Structure[i],
                Cols = Structure[i],
                Size = Structure[i] * Structure[i],
                Offset = 0
            };

            if (Structure[i] > MaxGateSize) MaxGateSize = Structure[i];
        }
        int NodeCountNoInputOutput = NodeCount - Structure[Structure.Length - 1];
        NodeCountNoInputOutput *= Population;
        NodeCount *= Population;
        WeightCount *= Population;
        StateWeightCount *= Population;
        int GateSize = MaxGateSize * Population;

        return new LSTMGroup()
        {
            Population = Population,
            Structure = Structure,
            _Structure = new ComputeBuffer(Structure.Length, sizeof(int)),

            Inputs = new float[Structure[0]],
            Nodes = new float[NodeCountNoInputOutput],
            Outputs = new float[Structure[Structure.Length - 1]],
            _Inputs = new ComputeBuffer(Structure[0], sizeof(float)),
            _Nodes = NodeCountNoInputOutput == 0 ? new ComputeBuffer(1, sizeof(float)) : new ComputeBuffer(NodeCountNoInputOutput, sizeof(float)),
            _Outputs = new ComputeBuffer(Structure[Structure.Length - 1], sizeof(float)),

            CellStates = new float[NodeCount],
            _CellStates = new ComputeBuffer(NodeCount, sizeof(float)),

            InputWeights = new float[WeightCount],
            InputStateWeights = new float[StateWeightCount],
            InputBias = new float[NodeCount],
            _InputWeights = new ComputeBuffer(WeightCount, sizeof(float)),
            _InputStateWeights = new ComputeBuffer(StateWeightCount, sizeof(float)),
            _InputBias = new ComputeBuffer(NodeCount, sizeof(float)),

            InputModulationWeights = new float[WeightCount],
            InputModulationStateWeights = new float[StateWeightCount],
            InputModulationBias = new float[NodeCount],
            _InputModulationWeights = new ComputeBuffer(WeightCount, sizeof(float)),
            _InputModulationStateWeights = new ComputeBuffer(StateWeightCount, sizeof(float)),
            _InputModulationBias = new ComputeBuffer(NodeCount, sizeof(float)),

            ForgetWeights = new float[WeightCount],
            ForgetStateWeights = new float[StateWeightCount],
            ForgetBias = new float[NodeCount],
            _ForgetWeights = new ComputeBuffer(WeightCount, sizeof(float)),
            _ForgetStateWeights = new ComputeBuffer(StateWeightCount, sizeof(float)),
            _ForgetBias = new ComputeBuffer(NodeCount, sizeof(float)),

            OutputWeights = new float[WeightCount],
            OutputStateWeights = new float[StateWeightCount],
            OutputBias = new float[NodeCount],
            _OutputWeights = new ComputeBuffer(WeightCount, sizeof(float)),
            _OutputStateWeights = new ComputeBuffer(StateWeightCount, sizeof(float)),
            _OutputBias = new ComputeBuffer(NodeCount, sizeof(float)),

            NodeCellInfo = NodeCellInfo,
            WeightBiasInfo = WeightBiasInfo,
            MaxGateSize = MaxGateSize,
            _NodeCellInfo = new ComputeBuffer(NodeCellInfo.Length, sizeof(int) * 4),
            _WeightBiasInfo = new ComputeBuffer(WeightBiasInfo.Length, sizeof(int) * 4),

            InputGate = new float[GateSize],
            InputModulationGate = new float[GateSize],
            ForgetGate = new float[GateSize],
            OutputGate = new float[GateSize],
            _InputGate = new ComputeBuffer(GateSize, sizeof(float)),
            _InputModulationGate = new ComputeBuffer(GateSize, sizeof(float)),
            _ForgetGate = new ComputeBuffer(GateSize, sizeof(float)),
            _OutputGate = new ComputeBuffer(GateSize, sizeof(float))
        };
    }

    public static void AssignLSTMGroupToShader(LSTMGroup Group, ComputeShader Compute)
    {
        Compute.SetInt("StructureLength", Group.Structure.Length);
        Compute.SetBuffer(0, "Structure", Group._Structure);
        Compute.SetInt("Population", Group.Population);

        Compute.SetBuffer(0, "Inputs", Group._Inputs);
        Compute.SetInt("NodeCount", Group.Nodes.Length);
        Compute.SetBuffer(0, "Nodes", Group._Nodes);
        Compute.SetBuffer(0, "Outputs", Group._Outputs);

        Compute.SetBuffer(0, "CellStates", Group._CellStates);

        Compute.SetBuffer(0, "InputWeights", Group._InputWeights);
        Compute.SetBuffer(0, "InputStateWeights", Group._InputStateWeights);
        Compute.SetBuffer(0, "InputBias", Group._InputBias);

        Compute.SetBuffer(0, "InputModulationWeights", Group._InputModulationWeights);
        Compute.SetBuffer(0, "InputModulationStateWeights", Group._InputModulationStateWeights);
        Compute.SetBuffer(0, "InputModulationBias", Group._InputModulationBias);

        Compute.SetBuffer(0, "ForgetWeights", Group._ForgetWeights);
        Compute.SetBuffer(0, "ForgetStateWeights", Group._ForgetStateWeights);
        Compute.SetBuffer(0, "ForgetBias", Group._ForgetBias);

        Compute.SetBuffer(0, "OutputWeights", Group._OutputWeights);
        Compute.SetBuffer(0, "OutputStateWeights", Group._OutputStateWeights);
        Compute.SetBuffer(0, "OutputBias", Group._OutputBias);

        Compute.SetBuffer(0, "NodeCellInfo", Group._NodeCellInfo);
        Compute.SetBuffer(0, "WeightBiasInfo", Group._WeightBiasInfo);
        Compute.SetInt("MaxGateSize", Group.MaxGateSize);

        Compute.SetBuffer(0, "InputGate", Group._InputGate);
        Compute.SetBuffer(0, "InputModulationGate", Group._InputModulationGate);
        Compute.SetBuffer(0, "ForgetGate", Group._ForgetGate);
        Compute.SetBuffer(0, "OutputGate", Group._OutputGate);
    }

    public static void DisposeGroup(LSTMGroup Group)
    {
        Group._Structure?.Dispose();
        
        Group._Inputs?.Dispose();
        Group._Nodes?.Dispose();
        Group._Outputs?.Dispose();

        Group._CellStates?.Dispose();

        Group._InputWeights?.Dispose();
        Group._InputStateWeights?.Dispose();
        Group._InputBias?.Dispose();

        Group._InputModulationWeights?.Dispose();
        Group._InputModulationStateWeights?.Dispose();
        Group._InputModulationBias?.Dispose();

        Group._ForgetWeights?.Dispose();
        Group._ForgetStateWeights?.Dispose();
        Group._ForgetBias?.Dispose();

        Group._OutputWeights?.Dispose();
        Group._OutputStateWeights?.Dispose();
        Group._OutputBias?.Dispose();

        Group._NodeCellInfo?.Dispose();
        Group._WeightBiasInfo?.Dispose();

        Group._InputGate?.Dispose();
        Group._InputModulationGate?.Dispose();
        Group._ForgetGate?.Dispose();
        Group._OutputGate?.Dispose();
    }
}
