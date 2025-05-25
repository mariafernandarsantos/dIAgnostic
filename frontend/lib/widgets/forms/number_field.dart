import 'package:flutter/material.dart';
import '../../core/utils/validators.dart';
import '../common/custom_text_field.dart';

class NumberField extends StatelessWidget {
  final String label;
  final TextEditingController controller;
  final bool isDecimal;
  final bool isRequired;

  const NumberField({
    super.key,
    required this.label,
    required this.controller,
    this.isDecimal = false,
    this.isRequired = true,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: CustomTextField(
        hintText: label,
        controller: controller,
        keyboardType: TextInputType.numberWithOptions(decimal: isDecimal),
        validator: isRequired
            ? (value) => Validators.validateNumber(value, label)
            : null,
      ),
    );
  }
}