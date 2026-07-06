import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function Dashboard({ evaluations }) {
  if (!evaluations || evaluations.length === 0) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
        <h2>Evaluation Dashboard</h2>
        <p>No evaluation data yet. Upload a PDF and ask some queries to generate metrics.</p>
      </div>
    );
  }

  // Format data for Recharts
  const chartData = evaluations.map((ev, index) => ({
    name: `Q${index + 1}`,
    Faithfulness: ev.faithfulness,
    Relevance: ev.answer_relevance,
    query: ev.query,
  }));

  return (
    <div style={{ padding: '2rem', width: '100%', height: '100%', overflowY: 'auto' }}>
      <h2 style={{ color: '#f1f5f9', marginBottom: '8px' }}>Ragas Evaluation Dashboard</h2>
      <p style={{ marginBottom: '2rem', color: '#94a3b8' }}>
        Metrics generated using the Ragas framework (Faithfulness and Answer Relevance).
      </p>
      
      <div style={{ width: '100%', height: 400, backgroundColor: '#1e293b', padding: '1rem', borderRadius: '8px', border: '1px solid #334155' }}>
        <ResponsiveContainer>
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" domain={[0, 1]} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', borderRadius: '6px' }}
              labelStyle={{ color: '#38bdf8', fontWeight: 'bold' }}
              formatter={(value, name) => [`${value.toFixed(2)}`, name]}
            />
            <Legend wrapperStyle={{ color: '#cbd5e1', paddingTop: '10px' }} />
            <Bar dataKey="Faithfulness" fill="#3b82f6" name="Faithfulness" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Relevance" fill="#10b981" name="Answer Relevance" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '2rem' }}>
        <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>Evaluation Details</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ backgroundColor: '#1e293b', textAlign: 'left' }}>
              <th style={{ padding: '12px', borderBottom: '2px solid #334155', color: '#f1f5f9' }}>Query</th>
              <th style={{ padding: '12px', borderBottom: '2px solid #334155', color: '#f1f5f9' }}>Faithfulness</th>
              <th style={{ padding: '12px', borderBottom: '2px solid #334155', color: '#f1f5f9' }}>Relevance</th>
            </tr>
          </thead>
          <tbody>
            {evaluations.map((ev, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #334155', backgroundColor: i % 2 === 0 ? 'transparent' : '#0f172a' }}>
                <td style={{ padding: '12px', color: '#cbd5e1' }}>{ev.query}</td>
                <td style={{ padding: '12px', color: ev.faithfulness > 0.7 ? '#10b981' : '#f59e0b', fontWeight: 'bold' }}>
                  {ev.faithfulness.toFixed(2)}
                </td>
                <td style={{ padding: '12px', color: ev.answer_relevance > 0.7 ? '#10b981' : '#f59e0b', fontWeight: 'bold' }}>
                  {ev.answer_relevance.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
