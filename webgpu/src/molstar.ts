/**
 * Mol* viewer wrapper: init, load PDB from string, color by pLDDT.
 */

import { DefaultPluginSpec } from 'molstar/lib/mol-plugin/spec';
import { PluginContext } from 'molstar/lib/mol-plugin/context';
import { PluginConfig } from 'molstar/lib/mol-plugin/config';

let plugin: PluginContext | null = null;

/**
 * Initialize the Mol* plugin in the given container element.
 */
export async function initViewer(container: HTMLDivElement): Promise<void> {
  if (plugin) return;

  const spec = DefaultPluginSpec();
  spec.config = spec.config || [];
  spec.config.push([PluginConfig.Viewport.ShowExpand, false]);
  spec.config.push([PluginConfig.Viewport.ShowControls, false]);
  spec.config.push([PluginConfig.Viewport.ShowSettings, false]);
  spec.config.push([PluginConfig.Viewport.ShowSelectionMode, false]);
  spec.config.push([PluginConfig.Viewport.ShowAnimation, false]);

  plugin = new PluginContext(spec);
  await plugin.init();

  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  container.appendChild(canvas);

  if (!plugin.initViewer(canvas, container as HTMLDivElement)) {
    console.warn('Mol* viewer init returned false');
  }
}

/**
 * Load a PDB string into the viewer, colored by B-factor (pLDDT).
 */
export async function loadStructure(pdbString: string): Promise<void> {
  if (!plugin) throw new Error('Mol* viewer not initialized');

  plugin.clear();

  const data = await plugin.builders.data.rawData({
    data: pdbString,
    label: 'RhoFold+ prediction',
  });

  const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');

  await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default', {
    showUnitcell: false,
    representationPreset: 'auto',
  });

  // Apply uncertainty coloring (B-factor = pLDDT * 100)
  try {
    await plugin.dataTransaction(async () => {
      for (const s of plugin!.managers.structure.hierarchy.current.structures) {
        await plugin!.managers.structure.component.updateRepresentationsTheme(
          s.components,
          { color: 'uncertainty' as any }
        );
      }
    });
  } catch (e) {
    console.warn('Could not apply uncertainty color theme:', e);
  }

  plugin.managers.camera.reset();
}

/**
 * Dispose the viewer and free resources.
 */
export function disposeViewer(): void {
  if (plugin) {
    plugin.dispose();
    plugin = null;
  }
}
