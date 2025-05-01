import 'package:flutter/material.dart';

class ResearchCard extends StatelessWidget {
  final Map<String, dynamic> research;
  final VoidCallback? onTap;
  
  const ResearchCard({
    Key? key,
    required this.research,
    this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16.0),
      elevation: 3,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: _getRelevanceColor(research['relevance'] ?? 'medium'),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      (research['relevance'] ?? 'medium').toUpperCase(),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  const Spacer(),
                  Text(
                    research['date'] ?? 'Recent',
                    style: TextStyle(
                      color: Colors.grey[600],
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                research['title'] ?? 'Untitled Research',
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                research['summary'] ?? 'No summary available',
                style: const TextStyle(fontSize: 14),
              ),
              const SizedBox(height: 12),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      'Source: ${research['source'] ?? 'Unknown'}',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[700],
                        fontStyle: FontStyle.italic,
                      ),
                    ),
                  ),
                  if (onTap != null)
                    Icon(
                      Icons.open_in_new,
                      size: 16,
                      color: Theme.of(context).primaryColor,
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Color _getRelevanceColor(String relevance) {
    switch (relevance.toLowerCase()) {
      case 'high':
        return Colors.green;
      case 'medium':
        return Colors.orange;
      case 'low':
        return Colors.blue;
      default:
        return Colors.grey;
    }
  }
}