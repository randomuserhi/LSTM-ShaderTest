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

        public float[] Inputs;
        public float[] Nodes;
        public float[] Outputs;
        public ComputeBuffer _Inputs;
        public ComputeBuffer _Outputs;
        public ComputeBuffer _Nodes;

        public float[] CellStates;
        public ComputeBuffer _CellStates;

        public float[] WeightsBiases;
        public ComputeBuffer _WeightsBiases;

        public MatrixInfo[] NodeCellInfo;
        public MatrixInfo[] WeightBiasInfo;
        public int MaxGateSize;
        public ComputeBuffer _NodeCellInfo;
        public ComputeBuffer _WeightBiasInfo;

        public float[] Gates;
        public ComputeBuffer _Gates;

        public void Initialize()
        {
            _Inputs.SetData(Inputs);
            _Outputs.SetData(Outputs);
            _Nodes.SetData(Nodes);
            _CellStates.SetData(CellStates);
            _WeightsBiases.SetData(WeightsBiases);
            _NodeCellInfo.SetData(NodeCellInfo);
            _WeightBiasInfo.SetData(WeightBiasInfo);
            _Gates.SetData(Gates);
        }
        public void SetWeightBiasData()
        {
            _WeightsBiases.SetData(WeightsBiases);
        }
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
        int TotalWeightBiasCount = (WeightCount + StateWeightCount + NodeCount) * 4;
        int TotalGateSize = GateSize * 4;

        int InputSize = Structure[0] * Population;
        int OutputSize = Structure[Structure.Length - 1] * Population;

        return new LSTMGroup()
        {
            Population = Population,
            Structure = Structure,

            Inputs = new float[InputSize],
            Nodes = new float[NodeCountNoInputOutput],
            Outputs = new float[OutputSize],
            _Inputs = new ComputeBuffer(InputSize, sizeof(float)),
            _Nodes = NodeCountNoInputOutput == 0 ? new ComputeBuffer(1, sizeof(float)) : new ComputeBuffer(NodeCountNoInputOutput, sizeof(float)),
            _Outputs = new ComputeBuffer(OutputSize, sizeof(float)),

            CellStates = new float[NodeCount],
            _CellStates = new ComputeBuffer(NodeCount, sizeof(float)),

            WeightsBiases = new float[TotalWeightBiasCount],
            _WeightsBiases = new ComputeBuffer(TotalWeightBiasCount, sizeof(float)),

            NodeCellInfo = NodeCellInfo,
            WeightBiasInfo = WeightBiasInfo,
            MaxGateSize = MaxGateSize,
            _NodeCellInfo = new ComputeBuffer(NodeCellInfo.Length, sizeof(int) * 4),
            _WeightBiasInfo = new ComputeBuffer(WeightBiasInfo.Length, sizeof(int) * 4),

            Gates = new float[TotalGateSize],
            _Gates = new ComputeBuffer(TotalGateSize, sizeof(float))
        };
    }

    public static void AssignLSTMGroupToShader(LSTMGroup Group, ComputeShader Compute)
    {
        Compute.SetInt("StructureLength", Group.Structure.Length);
        Compute.SetInt("Population", Group.Population);

        Compute.SetBuffer(0, "Inputs", Group._Inputs);
        Compute.SetBuffer(0, "Nodes", Group._Nodes);
        Compute.SetBuffer(0, "Outputs", Group._Outputs);

        Compute.SetBuffer(0, "CellStates", Group._CellStates);

        Compute.SetBuffer(0, "WeightsBiases", Group._WeightsBiases);

        Compute.SetBuffer(0, "NodeCellInfo", Group._NodeCellInfo);
        Compute.SetBuffer(0, "WeightBiasInfo", Group._WeightBiasInfo);
        Compute.SetInt("MaxGateSize", Group.MaxGateSize);
        Compute.SetInt("TotalGateStride", Group.MaxGateSize * 4);

        Compute.SetBuffer(0, "Gates", Group._Gates);
    }

    public static float[] FeedForward(float[] Inputs, LSTMGroup Group, ComputeShader Compute)
    {
        for (int i = 0; i < Inputs.Length; i++)
        {
            Group.Inputs[i] = Inputs[i];
        }
        Group._Inputs.SetData(Group.Inputs);
        Compute.Dispatch(0, Group.Population, 1, 1);
        Group._Outputs.GetData(Group.Outputs);
        return Group.Outputs;
    }

    public static float[] FeedForward(float[] Inputs, int Count, LSTMGroup Group, ComputeShader Compute)
    {
        for (int i = 0; i < Inputs.Length; i++)
        {
            Group.Inputs[i] = Inputs[i];
        }
        Group._Inputs.SetData(Group.Inputs);
        Compute.Dispatch(0, Count, 1, 1);
        Group._Outputs.GetData(Group.Outputs);
        return Group.Outputs;
    }

    public static void DisposeGroup(LSTMGroup Group)
    {
        Group._Inputs?.Dispose();
        Group._Nodes?.Dispose();
        Group._Outputs?.Dispose();

        Group._CellStates?.Dispose();

        Group._WeightsBiases?.Dispose();

        Group._NodeCellInfo?.Dispose();
        Group._WeightBiasInfo?.Dispose();

        Group._Gates?.Dispose();
    }
}
