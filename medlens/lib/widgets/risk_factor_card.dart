import 'package:flutter/material.dart';

class RiskFactorCard extends StatelessWidget {
  final Map<String, dynamic> riskFactor;
  
  const RiskFactorCard({
    Key? key,
    required this.riskFactor,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16.0),
      elevation: 3,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(
          color: _getImpactColor(riskFactor['impact'] ?? 'medium').withOpacity(0.5),
          width: 1.5,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.warning_rounded,
                  color: _getImpactColor(riskFactor['impact'] ?? 'medium'),
                  size: 24,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    riskFactor['factor'] ?? 'Unknown Risk Factor',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: _getImpactColor(riskFactor['impact'] ?? 'medium'),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    (riskFactor['impact'] ?? 'medium').toUpperCase(),
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              riskFactor['description'] ?? 'No description available',
              style: const TextStyle(fontSize: 14),
            ),
            const SizedBox(height: 16),
            if (_hasRecommendations())
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Recommendations:',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  ..._buildRecommendations(),
                ],
              ),
          ],
        ),
      ),
    );
  }
  
  bool _hasRecommendations() {
    return riskFactor['recommendations'] != null && 
           riskFactor['recommendations'] is List && 
           (riskFactor['recommendations'] as List).isNotEmpty;
  }
  
  List<Widget> _buildRecommendations() {
    final recommendations = riskFactor['recommendations'] as List;
    return recommendations.map<Widget>((rec) => 
      Padding(
        padding: const EdgeInsets.only(bottom: 6.0),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(
              Icons.arrow_right,
              size: 20,
              color: Colors.blueGrey,
            ),
            const SizedBox(width: 4),
            Expanded(
              child: Text(rec.toString()),
            ),
          ],
        ),
      )
    ).toList();
  }
  
  Color _getImpactColor(String impact) {
    switch (impact.toLowerCase()) {
      case 'high':
        return Colors.red[700]!;
      case 'medium':
        return Colors.orange[700]!;
      case 'low':
        return Colors.blue[700]!;
      default:
        return Colors.grey[700]!;
    }
  }
}